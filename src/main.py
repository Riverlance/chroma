'''
Copyright (c) 2025 Ibict Authors. All rights reserved.
'''



'''
To load parameters from .env, add the following content to the bottom of your `venv/bin/activate` (`nano venv/bin/activate`):
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi
'''

import chromadb
import ijson
import time
from pathlib import Path
# import traceback # Debugging (t.print_stack())

PROJECT_ROOT = Path(__file__).parent.parent





# Error codes
RAG_ERROR                = 0
RAG_ERROR_NOJSONFILEPATH = 1
RAG_ERROR_NOCLIENT       = 1
RAG_ERROR_NOCOLLECTION   = 2

class RagHandler:
  # Error messages
  __ERROR_MESSAGES = {
    RAG_ERROR               : "Sorry, not possible.",
    RAG_ERROR_NOJSONFILEPATH: "JSON file path is not provided.",
    RAG_ERROR_NOCLIENT      : "Client is not created yet. Call create_client(...) first.",
    RAG_ERROR_NOCOLLECTION  : "Collection is not created yet. Call create_collection(...) first.",
  }



  # region MARK: Self

  def __init__(self, json_filepath: str = None, client_path: str = None, collection_name: str = None, embedding_function: object = None):
    self.json_filepath      = json_filepath
    self.client_path        = client_path
    self.collection_name    = collection_name
    self.embedding_function = embedding_function

    self.client     = None
    self.collection = None

    # JSON parsing data
    self.unique_ids = [ ]
    self.metadatas  = [ ]
    self.documents  = [ ]

    # Init client
    if client_path:
      self.create_client()

    # Init empty collection
    if collection_name:
      self.create_collection()

  def has_data(self):
    '''
    Check if collection has data.

    Returns:
      bool: `True` if collection has data, `False` otherwise.
    '''
    return self.collection and self.collection.count()

  @staticmethod
  def error(id):
    '''
    Get error message.

    Args:
      id (int): Error id.

    Returns:
      str: Error message.
    '''

    return RagHandler.__ERROR_MESSAGES[id]

  # endregion



  # region MARK: JSON parsing

  def __parse_object(self, obj: dict) -> tuple:
    '''
    Parse a single object from a JSON file.

    Args:
      obj (dict): A single object from a JSON file.

    Returns:
      tuple: A tuple of metadatas and documents.
    '''

    # Note:
    # There is a problem that obj.get('INSTITUICAO', '') only uses the default value if the key doesn't exist in json.
    # But if the key explicitly exists with a `null` value (which becomes `None` in Python), it returns the explicit `None`, instead of the default value.
    # One work around for this, is to force the fallback manually. For example: `obj.get('INSTITUICAO') or ''`.
    # Therefore, if the value is `None`, `''` or another "falsy" value, then `''` will be used instead.

    group_id             = int(obj.get('group_id') or 0)
    instituicao          = (obj.get('INSTITUICAO') or '').strip()
    biblioteca           = (obj.get('BIBLIOTECA_NOME') or '').strip()
    editora              = (obj.get('NOME_EDITORA') or '').strip()
    area_conhecimento    = (obj.get('AREA_CONHECIMENTO') or '').strip()
    assuntos_controlados = (obj.get('SPINES') or '').strip()
    termo_livre          = (obj.get('TERMO_LIVRE') or '').strip()

    # 5 documents per object
    # Document #1 - 'TITULO_PUBLICACAO'
    doc_1 = (obj.get('TITULO_PUBLICACAO') or '').strip()
    # Document #2 - 'TITULO_RELACIONADO'
    doc_2 = (obj.get('TITULO_RELACIONADO') or '').strip()
    # Document #3 - 'COLECAO'
    doc_3 = (obj.get('COLECAO') or '').strip()
    # Document #4 - 'COMENTARIO'
    doc_4 = (obj.get('COMENTARIO') or '').strip()
    # Document #5 - 'CONTEXT' (metadatas as document)
    doc_5 = f'Instituição: "{instituicao}"; Biblioteca: "{biblioteca}"; Editora: "{editora}"; Área do Conhecimento: "{area_conhecimento}"; Assuntos Controlados: "{assuntos_controlados}"; Termo Livre: "{termo_livre}"; Título da Publicação: "{doc_1}"; Título Relacionado: "{doc_2}"; Comentário: "{doc_4}"'

    # Metadatas
    metadatas = {
      'group_id'          : str(group_id),
      'COD_CCN_PUBLICACAO': (obj.get('COD_CCN_PUBLICACAO') or '').strip(),
      'INSTITUICAO'       : instituicao,
      'BIBLIOTECA_NOME'   : biblioteca,
      'NOME_EDITORA'      : editora,
      'AREA_CONHECIMENTO' : area_conhecimento,
      'SPINES'            : assuntos_controlados,
      'TERMO_LIVRE'       : termo_livre,

      # Documents also as metadatas (except by the context document)
      'TITULO_PUBLICACAO' : doc_1,
      'TITULO_RELACIONADO': doc_2,
      'COLECAO'           : doc_3,
      'COMENTARIO'        : doc_4
    }

    # Remove empty documents
    docs = [doc.strip() for doc in (doc_1, doc_2) if doc.strip()]

    return metadatas, *docs

  def __parse_json_file(self, limit: int = None):
    '''
    Parse a JSON file in streaming mode (reads without loading entire file into memory).
    It also yields (produces) documents to the generator.

    Args:
      limit (int, optional): The maximum number of objects to parse.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    parsed_amount = 0
    unique_id     = 0
    json_info     = self.get_json_file_info()

    print(f">> Starting to parse '{json_info[0].name}' ({json_info[1]:.2f} MB)")

    with open(self.json_filepath, 'rb') as file:
      # Parse streaming
      '''
      json format example:
      [
        { ... }, # Item 1
        { ... }, # Item 2
        { ... }  # Item 3
      ]
      '''
      file.seek(0) # Reset file cursor
      for obj in ijson.items(file, prefix = 'item'):
        parsed_amount  += 1
        obj['group_id'] = parsed_amount # Id for the group of documents
        data            = self.__parse_object(obj)

        # Needs at least metadatas and 1 document
        if len(data) < 2:
          continue

        metadatas  = data[0]
        docs       = data[1:]
        unique_ids = [ ]

        # Generate unique ids
        for _ in range(len(docs)):
          unique_id += 1
          unique_ids.append(str(unique_id))

        # Deliver id, metadatas and documents to the caller (generator)
        yield unique_ids, metadatas, docs

        # Display progress
        if parsed_amount % 1000 == 0:
          print(f"> {parsed_amount:,} objects parsed.")

        # Stop if limit is reached
        if limit and parsed_amount >= limit:
          break

  def load(self, *a, **k):
    '''
    Parse a JSON file and store its data in internal lists.
    Same parameters as __parse_json_file.

    Args:
      *a (tuple): Positional arguments.
      **k (dict): Keyword arguments.
    '''

    for unique_ids, metadatas, docs in self.__parse_json_file(*a, **k):
      # Store unique ids, metadatas and documents
      for id, doc in zip(unique_ids, docs):
        self.unique_ids.append(id)
        self.metadatas.append(metadatas)
        self.documents.append(doc)

  def get_json_file_info(self):
    '''
    Get information about a JSON file.

    Returns:
      path (Path): The path to the JSON file.
      mb_size (float): The size of the JSON file in megabytes.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    path    = Path(self.json_filepath)
    mb_size = round(path.stat().st_size / (1024 ** 2), 2)

    return path, mb_size

  def print_json_file_data(self, *a, **k):
    '''
    Parse a JSON file and print its data.
    Same parameters as __parse_json_file.

    Args:
      *a (tuple): Positional arguments.
      **k (dict): Keyword arguments.
    '''

    for unique_ids, metadatas, docs in self.__parse_json_file(*a, **k):
      print(f">> Found new document of group #{metadatas.get('group_id') or 0}:\n")

      print(f"> Metadatas:\n{metadatas}\n")

      for id, doc in zip(unique_ids, docs):
        print(f"> Document #{id}\n{doc}\n")

      print('--- ' * 7 + '\n')

  # endregion



  # region MARK: Chroma

  # region MARK: VectorDB

  def create_client(self):
    '''
    Create a Chroma client.
    '''

    # Create a Chroma persistent client
    self.client = chromadb.PersistentClient(path = self.client_path, settings = chromadb.config.Settings(anonymized_telemetry = False))

    print(f">> Client has been created successfully at '{self.client_path}'")

  def create_collection(self):
    '''
    Create a Chroma collection.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    # Get existing collection
    try:
      self.collection = self.client.get_collection(name = self.collection_name)
      print(f">> Collection '{self.collection_name}' has been loaded successfully with {self.collection.count()} items")

    # Collection doesn't exist
    except chromadb.errors.NotFoundError:
      self.collection = self.client.create_collection(name = self.collection_name, embedding_function = self.embedding_function)
      print(f">> Collection '{self.collection_name}' has been created successfully")

  def delete_collection(self):
    '''
    Delete the Chroma collection.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    for collection in self.client.list_collections():
      try:
        self.client.delete_collection(name = collection.name)
      except:
        pass

  def create_vectordb(self):
    '''
    Create a Chroma vector database.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)
    assert self.collection, RagHandler.error(RAG_ERROR_NOCOLLECTION)

    print(f">> Starting to create the vector database")

    batch_range   = 100 # Number of objects to add at once per batch
    saved_amount  = 0
    total_amount  = len(self.unique_ids)
    time_begin    = time.time()
    time_previous = time_begin # Time of previous batch execution
    time_current  = time_begin # Time of current batch execution

    # If empty, stop
    if total_amount == 0:
      print("> Nothing to save.")
      return

    for i in range(0, total_amount, batch_range):
      index_end = min(i + batch_range, total_amount)

      # Add content to the collection
      try:
        self.collection.add(ids = self.unique_ids[i:index_end], metadatas = self.metadatas[i:index_end], documents = self.documents[i:index_end])

        saved_amount += index_end - i
      except Exception as err:
        print(f"> Error in batch from {i} to {index_end}: {err}")

      # Batch progress

      time_current = time.time()
      if time_current - time_previous > 1:
        time_previous  = time_current
        progress_ratio = (saved_amount / total_amount) if total_amount else 0

        # Complete already, then break
        if progress_ratio >= 1:
          break

        # Print actual progress
        print(f"> Saved {saved_amount:,} objects in {round(time_current - time_begin, 2):.2f} seconds ({(progress_ratio * 100):.2f}%)")

    # Print last progress
    real_saved_amount = self.collection.count()
    saved_amount_text = f"{real_saved_amount:,} of {saved_amount:,}" if real_saved_amount < saved_amount else f"{saved_amount:,}"
    print(f"> Saved {saved_amount_text} objects in {round(time_current - time_begin, 2):.2f} seconds.")

  # endregion



  # region MARK: Search

  def search(self, query_text: str, n_results: int = 10):
    '''
    Search for relevant documents in the Chroma collection.
    It needs a vector database to be already created and filled with the embeddings of the documents.

    Args:
      query_text (str): The query text to search for.
      n_results (int, optional): The number of results to return.
    '''

    assert self.collection, RagHandler.error(RAG_ERROR_NOCOLLECTION)

    print(f"\n>> Requested query: \"{query_text}\"")

    # Search for relevant documents and their metadatas
    results = self.collection.query(query_texts = [query_text], n_results = n_results)

    # Get the metadatas and documents
    ids       = results['ids'][0]
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]

    # Display the results

    # Found results
    if ids and metadatas and documents:
      print(f">> Found {len(documents)} relevant results:\n")

      for i, (id, metadata, document) in enumerate(zip(ids, metadatas, documents), 1):
        print(f"> Result #{i} - Document #{id}\n")
        print(f"> Metadatas:\n{metadata}\n")
        print(f"> Document (255 chars only):\n{document[:255]}{'...' if len(document) > 255 else ''}\n")
        print('--- ' * 7 + '\n')

    # No results found
    else:
      print(">> No results found.")

  def init_search_terminal_mode(self, n_results: int = 10):
    '''
    It uses the search method to search in the terminal for relevant documents in the Chroma collection.

    Args:
      n_results (int, optional): The number of results to return. Defaults to 10.
    '''
    print(">> RAG Search ~")
    print('> Type "quit" to exit.\n\n')

    while True:
      query = input("\nType your question:").strip()

      if query.lower() in ['quit', 'q', 'exit', 'sair']:
        print(">> Ending session.")
        break

      if query:
        self.search(query_text = query, n_results = n_results)
      else:
        print(">> Type a valid question.")

  # endregion

  # endregion





# Test, according to the Chroma docs
def testChromaClient():
  '''
  See: https://python.langchain.com/docs/modules/data_connection/document_transformers/chroma.html
  '''

  # Create a client
  client = chromadb.Client()

  # Create a collection
  collection = client.create_collection(name = "my_collection")

  # Add some data
  collection.upsert( # collection.add( # Use upsert, instead of add, to avoid adding the same documents every time
    ids = ['id1', 'id2'],
    documents = [
      "This is a document about pineapple",
      "This is a document about oranges",
    ]
  )

  results = collection.query(
    query_texts = ["This is a query document about hawaii"], # Chroma will embed this for you
    n_results = 2, # How many results to return (10 by default)
  )

  # Show the results
  print(results)
  '''
  # query_texts = ["This is a query document about hawaii"]
  {
    'documents': [[
      'This is a document about pineapple',
      'This is a document about oranges'
    ]],
    'ids': [['id1', 'id2']],
    'distances': [[1.0404009819030762, 1.2430799007415771]],
    'uris': None,
    'data': None,
    'metadatas': [[None, None]],
    'embeddings': None,
    'included': ['metadatas', 'documents', 'distances'],
  }
  '''
  '''
  # query_texts = ["This is a query document about florida"]
  {
    'documents': [[
      'This is a document about oranges',
      'This is a document about pineapple'
    ]],
    'ids': [['id2', 'id1']],
    'distances': [[1.1462137699127197, 1.3015384674072266]],
    'uris': None,
    'data': None,
    'metadatas': [[None, None]],
    'embeddings': None,
    'included': ['metadatas', 'documents', 'distances'],
  }
  '''





if __name__ == "__main__":
  # testChromaClient()



  '''
  Parse a JSON file and print its data
  '''

  # rag_parser = RagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')

  # # Get info about a JSON file
  # print(rag_parser.get_json_file_info())

  # # Parse a JSON file in streaming mode (see print_json_file_data)
  # rag_parser.print_json_file_data(limit = 10)

  # # Parse a JSON file and print its data of internal lists
  # rag_parser.load(limit = 10)
  # print()
  # print(rag_parser.unique_ids)
  # print()
  # print(rag_parser.metadatas)
  # print()
  # print(rag_parser.documents)



  '''
  Parse a JSON file into a vector database
  '''

  # rag_vectordb = RagHandler(json_filepath      = f'{PROJECT_ROOT}/data/db.json',
  #                           client_path        = f'{PROJECT_ROOT}/output',
  #                           collection_name    = 'data',
  #                           embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'paraphrase-multilingual-MiniLM-L12-v2'))

  # # Parse a JSON file and save its data as a vector database
  # rag_vectordb.load(limit = 1000)
  # rag_vectordb.create_vectordb()



  '''
  Search in a vector database
  '''

  # rag_search = RagHandler(client_path     = f'{PROJECT_ROOT}/output',
  #                         collection_name = 'data')

  # # Search "Journal" in the vector database
  # rag_search.search(query_text = "Me mostre publicações de psicologia", n_results = 10)

  # # # Init search in terminal mode
  # rag_search.init_search_terminal_mode(n_results = 10)
