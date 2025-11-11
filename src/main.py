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
import pathlib
import time
# import traceback # Debugging (t.print_stack())

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

JSON_PARSING_PRINT_CYCLETIME    = 1 # seconds
VECTORDB_SAVING_PRINT_CYCLETIME = 1 # seconds





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

  def __init__(self, json_filepath: str = None, collection_name: str = None, embedding_function: object = None):
    self.json_filepath      = json_filepath
    self.collection_name    = collection_name
    self.embedding_function = embedding_function

    self.client     = None
    self.collection = None

    # JSON parsing data
    self.unique_ids        = [ ]
    self.metadatas         = [ ]
    self.documents         = [ ]
    self.empty_docs_amount = 0

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
    # Document #4 - 'COMENTARIO'
    doc_3 = (obj.get('COMENTARIO') or '').strip()
    # Document #5 - 'CONTEXT' (metadatas as document)
    doc_4 = f'Instituição: "{instituicao}"; Biblioteca: "{biblioteca}"; Editora: "{editora}"; Área do Conhecimento: "{area_conhecimento}"; Assuntos Controlados: "{assuntos_controlados}"; Termo Livre: "{termo_livre}"; Título da Publicação: "{doc_1}"; Título Relacionado: "{doc_2}"; Comentário: "{doc_3}"'

    # Metadatas
    metadatas = {
      'group_id'          : str(group_id),
      'COD_CCN_PUBLICACAO': (obj.get('COD_CCN_PUBLICACAO') or '').strip(),
      'COLECAO'           : (obj.get('COLECAO') or '').strip(),
      'INSTITUICAO'       : instituicao,
      'BIBLIOTECA_NOME'   : biblioteca,
      'NOME_EDITORA'      : editora,
      'AREA_CONHECIMENTO' : area_conhecimento,
      'SPINES'            : assuntos_controlados,
      'TERMO_LIVRE'       : termo_livre,

      # Documents also as metadatas (except by the context document)
      'TITULO_PUBLICACAO' : doc_1,
      'TITULO_RELACIONADO': doc_2,
      'COMENTARIO'        : doc_3
    }

    # Remove empty documents
    docs              = [doc_1, doc_2, doc_3, doc_4]
    docs_amount       = len(docs)
    filled_docs       = [doc.strip() for doc in docs if doc.strip()]
    empty_docs_amount = docs_amount - len(filled_docs)

    return metadatas, empty_docs_amount, *filled_docs

  def __parse_json_file(self, limit: int = None):
    '''
    Parse a JSON file in streaming mode (reads without loading entire file into memory).
    It also yields (produces) documents to the generator.

    Args:
      limit (int, optional): The maximum number of objects to parse.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    global JSON_PARSING_PRINT_CYCLETIME

    parsed_amount = 0
    unique_id     = 0
    json_info     = self.get_json_file_info()
    time_begin    = time.time()
    time_previous = time_begin # Time of previous batch execution
    time_current  = time_begin # Time of current batch execution

    print(f">> Starting to parse '{json_info[0].name}' ({json_info[1]:.2f} MB)...")

    with open(self.json_filepath, 'rb') as file:
      # Parse streaming

      '''
      Expected json format:
      [
        { ... }, # Item 1
        { ... }, # Item 2
        { ... }  # Item 3
      ]
      '''

      # Reset file cursor
      file.seek(0)

      for obj in ijson.items(file, prefix = 'item'):
        parsed_amount    += 1
        obj['group_id']   = parsed_amount # Id for the group of documents
        data              = self.__parse_object(obj)
        data_len          = len(data)
        unique_ids        = [ ]
        metadatas         = data[0] if data_len > 0 else { }
        empty_docs_amount = data[1] if data_len > 1 else 0
        docs              = data[2:] if data_len > 2 else [ ]

        # Update empty documents amount
        self.empty_docs_amount += empty_docs_amount

        # Generate unique ids
        for _ in range(len(docs)):
          unique_id += 1
          unique_ids.append(str(unique_id))

        # Deliver id, metadatas and documents to the caller (generator)
        yield unique_ids, metadatas, docs

        # Display progress
        time_current = time.time()
        if time_current - time_previous > JSON_PARSING_PRINT_CYCLETIME:
          time_previous = time_current

          # Print actual progress
          time_elapsed = round(time_current - time_begin)
          print(f"> {parsed_amount:,} objects parsed in {time_elapsed} second{'' if time_elapsed < 2 else 's'}.")

        # Stop if limit is reached
        if limit and parsed_amount >= limit:
          break

    print(f"> {parsed_amount:,} objects parsed from '{json_info[0].name}' ({json_info[1]:.2f} MB) in {round(time_current - time_begin, 2):.2f} seconds.\n")

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
      path (pathlib.Path): The path to the JSON file.
      mb_size (float): The size of the JSON file in megabytes.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    path    = pathlib.Path(self.json_filepath)
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

  def create_client(self, path: str):
    '''
    Create a Chroma client.
    '''

    print(">> Creating client...")

    # Create a Chroma persistent client
    self.client = chromadb.PersistentClient(path = path, settings = chromadb.config.Settings(anonymized_telemetry = False))

    print(f"> Client has been created successfully with path as '{path}'.\n")

  def create_collection(self):
    '''
    Create a Chroma collection.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    print(">> Creating collection...")

    # Get existing collection
    try:
      self.collection = self.client.get_collection(name = self.collection_name)
      print(f"> Collection '{self.collection_name}' has been loaded successfully with {self.collection.count():,} items.\n")

    # Collection doesn't exist
    except chromadb.errors.NotFoundError:
      self.collection = self.client.create_collection(name = self.collection_name, embedding_function = self.embedding_function)
      print(f"> Collection '{self.collection_name}' has been created successfully.\n")

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

    global VECTORDB_SAVING_PRINT_CYCLETIME

    batch_range   = 100 # Number of objects to add at once per batch
    saved_amount  = 0
    total_amount  = len(self.unique_ids)
    time_begin    = time.time()
    time_previous = time_begin # Time of previous batch execution
    time_current  = time_begin # Time of current batch execution

    print(f">> Starting to create the vector database...")

    # If empty, stop
    if total_amount == 0:
      print("> Nothing to save.")
      return

    for i in range(0, total_amount, batch_range):
      index_end = min(i + batch_range, total_amount)

      # Add content to the collection
      try:
        # .upsert instead of .add to avoid adding the same documents every time
        self.collection.upsert(ids = self.unique_ids[i:index_end], metadatas = self.metadatas[i:index_end], documents = self.documents[i:index_end])

        saved_amount += index_end - i
      except Exception as err:
        print(f"> Error in batch from {i} to {index_end}: {err}")

      # Batch progress

      time_current = time.time()
      if time_current - time_previous > VECTORDB_SAVING_PRINT_CYCLETIME:
        time_previous  = time_current
        progress_ratio = (saved_amount / total_amount) if total_amount else 0

        # Complete already, then break
        if progress_ratio >= 1:
          break

        # Print actual progress
        time_elapsed = round(time_current - time_begin)
        print(f"> Saved {saved_amount:,} documents in {time_elapsed} second{'' if time_elapsed < 2 else 's'} ({(progress_ratio * 100):.2f}%).")

    # Print last progress
    print(f"> Saved {saved_amount:,} documents (of {saved_amount + self.empty_docs_amount:,}, ignoring {self.empty_docs_amount:,} empty documents) in {round(time_current - time_begin, 2):.2f} seconds.\n")

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
      n_results (int, optional): The number of results to return.
    '''

    print(">> RAG Search")

    while True:
      query = input("\nType your question (\"quit\" to exit): ").strip()

      if query.lower() in ['quit', 'q', 'exit', 'sair']:
        print(">> Ending session.")
        break

      if query:
        self.search(query_text = query, n_results = n_results)
      else:
        print(">> Type a valid question.")

  # endregion

  # endregion





if __name__ == "__main__":
  '''
  Parse a JSON file and print its data
  '''

  # # Get info about a JSON file
  # rag_parser = RagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')
  # print(rag_parser.get_json_file_info())

  # # Parse a JSON file in streaming mode (see print_json_file_data)
  # rag_parser = RagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')
  # rag_parser.print_json_file_data(limit = 10)

  # # Parse a JSON file and print its data of internal lists
  # rag_parser = RagHandler(json_filepath = f'{PROJECT_ROOT}/data/db.json')
  # rag_parser.load(limit = None)
  # print()
  # print(rag_parser.unique_ids)
  # print()
  # print(rag_parser.metadatas)
  # print()
  # print(rag_parser.documents)



  '''
  Parse a JSON file into a vector database
  '''

  # # Parse a JSON file and save its data as a vector database
  # print(">> Creating embedding function...")
  # embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'paraphrase-multilingual-MiniLM-L12-v2')
  # print("> Embedding function has been created successfully.\n")
  # rag_vectordb = RagHandler(json_filepath      = f'{PROJECT_ROOT}/data/db.json',
  #                           collection_name    = 'data',
  #                           embedding_function = embedding_function)
  # rag_vectordb.create_client(path = f'{PROJECT_ROOT}/output')
  # rag_vectordb.create_collection()
  # rag_vectordb.load(limit = 250)
  # rag_vectordb.create_vectordb()



  '''
  Search in a vector database
  '''

  # # Search "Psicologia" in the vector database
  # rag_search = RagHandler(collection_name = 'data')
  # rag_search.create_client(path = f'{PROJECT_ROOT}/output')
  # rag_search.create_collection()
  # rag_search.search(query_text = "Me mostre publicações de psicologia", n_results = 10)

  # # Init search in terminal mode
  # rag_search = RagHandler(collection_name = 'data')
  # rag_search.create_client(path = f'{PROJECT_ROOT}/output')
  # rag_search.create_collection()
  # rag_search.init_search_terminal_mode(n_results = 10)
