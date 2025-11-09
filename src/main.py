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

PROJECT_ROOT = Path(__file__).parent.parent





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

  def __init__(self, json_filepath: str = None, client_path: str = None, collection_name: str = None, embedding_function: object = None):
    self.json_filepath      = json_filepath
    self.client_path        = client_path
    self.collection_name    = collection_name
    self.embedding_function = embedding_function
    self.client             = None
    self.collection         = None

    if client_path:
      self.create_client(path = client_path)

    if collection_name:
      self.create_collection(name = collection_name, embedding_function = embedding_function)

    # Clear data
    self.clear_data()



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
    # The `ijson` parser returns None when the field is missing.
    # That's why I've used, for example, `obj['INSTITUICAO'] or ''`, instead of `obj.get('INSTITUICAO', '')`.
    instituicao          = obj['INSTITUICAO'] or ''
    biblioteca           = obj['BIBLIOTECA_NOME'] or ''
    editora              = obj['NOME_EDITORA'] or ''
    area_conhecimento    = obj['AREA_CONHECIMENTO'] or ''
    assuntos_controlados = obj['SPINES'] or ''
    termo_livre          = obj['TERMO_LIVRE'] or ''

    # Metadatas
    metadatas = {
      'id':                 obj['id'] or '',
      'COD_CCN_PUBLICACAO': obj['COD_CCN_PUBLICACAO'] or '',
      'INSTITUICAO':        instituicao,
      'BIBLIOTECA_NOME':    biblioteca,
      'NOME_EDITORA':       editora,
      'AREA_CONHECIMENTO':  area_conhecimento,
      'SPINES':             assuntos_controlados,
      'TERMO_LIVRE':        termo_livre,
    }

    # 5 documents per object
    doc_1 = dict(metadatas) # Copy metadatas
    doc_2 = dict(metadatas) # Copy metadatas
    doc_3 = dict(metadatas) # Copy metadatas
    doc_4 = dict(metadatas) # Copy metadatas
    doc_5 = dict(metadatas) # Copy metadatas

    # Document - 'TITULO_PUBLICACAO'
    doc_1['TITULO_PUBLICACAO'] = obj['TITULO_PUBLICACAO'] or ''

    # Document - 'TITULO_RELACIONADO'
    doc_2['TITULO_RELACIONADO'] = obj['TITULO_RELACIONADO'] or ''

    # Document - 'COLECAO'
    doc_3['COLECAO'] = obj['COLECAO'] or ''

    # Document - 'COMENTARIO'
    doc_4['COMENTARIO'] = obj['COMENTARIO'] or ''

    # Document - 'CONTEXT' (extra to join the metadata as a single string, keeping also the metadata structure)
    doc_5['CONTEXT'] = f"Instituição: {instituicao}; Biblioteca: {biblioteca}; Editora: {editora}; Área do Conhecimento: {area_conhecimento}; Assuntos Controlados: {assuntos_controlados}; Termo Livre: {termo_livre}"

    return metadatas, doc_1, doc_2, doc_3, doc_4, doc_5

  def __parse_json_file(self, limit: int = None):
    '''
    Parse a JSON file in streaming mode (reads without loading entire file into memory).
    It also yields (produces) documents to the generator.

    Args:
      limit (int, optional): The maximum number of objects to parse.
    '''

    assert self.json_filepath, RagHandler.error(RAG_ERROR_NOJSONFILEPATH)

    # Clear data
    self.clear_data()

    parsed_amount = 0
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
      file.seek(0)
      for obj in ijson.items(file, prefix = 'item'):
        parsed_amount += 1
        obj['id']      = str(parsed_amount) # Unique id for the group of documents
        data           = self.__parse_object(obj)
        metadata       = data[0]
        docs           = data[1:]

        # Deliver id, metadata and documents to the caller (generator)
        yield obj['id'], metadata, docs

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

    for id, metadatas, docs in rag.__parse_json_file(*a, **k):
      # Store ids, metadata and documents
      for doc in docs:
        self.ids.append(id)
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

    for id, metadata, docs in rag.__parse_json_file(*a, **k):
      doc_i = 0

      print(f">> Found new document #{id}")

      print(f"> Metadata:\n{metadata}\n")

      for doc in docs:
        doc_i += 1
        print(f"> Document #{doc_i}\n{doc}\n")

      print('-' * 10 + '\n')

  # endregion



  # region MARK: ChromaDB

  # region MARK: VectorDB

  def create_client(self, path: str = 'vectordb'):
    '''
    Create a ChromaDB client.

    Args:
      path (str, optional): The path to the VectorDB database.
    '''

    # Create a ChromaDB persistent client
    self.client = chromadb.PersistentClient(path = path)

    # Display a success message
    print(f">> ChromaDB client has been created successfully at '{path}'")

  def create_collection(self, name: str = 'data'):
    '''
    Create a ChromaDB collection.

    Args:
      name (str, optional): The name of the collection.
      embedding_function (object, optional): The embedding function to use.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    # Delete all collections
    self.delete_collection()

    # Create a collection
    self.collection = self.client.create_collection(name = name)

    # Display the number of objects in the collection
    print(f">> {self.collection.count()} objects found in the collection.")

  def delete_collection(self):
    '''
    Delete a ChromaDB collection.
    '''

    assert self.client, RagHandler.error(RAG_ERROR_NOCLIENT)

    for collection in self.client.list_collections():
      try:
        self.client.delete_collection(name = collection.name)
      except:
        pass

  def clear_collection(self):
    '''
    Clear collection in the ChromaDB database.
    '''

    assert self.collection, RagHandler.error(RAG_ERROR_NOCOLLECTION)

    # Clear data without deleting the collection
    self.collection.delete(where = { })

  # endregion



  # region MARK: Search

  def search(self, query_text: str, n_results: int = 10):
    '''
    Search for relevant documents in the ChromaDB collection.
    It needs a vector database to be already created and filled with the embeddings of the documents.

    Args:
      query_text (str): The query text to search for.
      n_results (int, optional): The number of results to return.
    '''

    assert self.collection, RagHandler.error(RAG_ERROR_NOCOLLECTION)

    print(f"\n>> Requested query: \"{query_text}\"\n\n")

    # Search for relevant documents and their metadatas
    results = self.collection.query(query_texts = [query_text], n_results = n_results)

    # Get the metadatas and documents
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]

    # Display the results

    # Found results
    if metadatas and documents:
      print(f">> Found {len(documents)} relevant results:\n")

      for i, (metadata, document) in enumerate(zip(metadatas, documents), 1):
        print(f"{i}")

        # todo - print data
        # print(f"{i}. {metadata['titulo']}")
        # print(f"Author: {metadata['autor']}")
        # print(f"Resume: {document[:200]}...")
        # print(f"ID: {metadata['id_original']}")
        print()

    # No results found
    else:
      print(">> No results found.")

  def init_search_terminal_mode(self, n_results: int = 10):
    '''
    It uses the search method to search in the terminal for relevant documents in the ChromaDB collection.

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



  # region MARK: Self

  def clear_data(self):
    '''
    Clear stored data.
    '''

    # Clear collection, if it exists
    if self.collection:
      self.clear_collection()

    # Clear internal lists
    self.ids       = []
    self.metadatas = []
    self.documents = []

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





# Test, according to the docs
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

  rag = RagHandler(json_filepath = f"{PROJECT_ROOT}/data/db.json")

  # Get info about a JSON file
  # print(rag.get_json_file_info())

  # Parse a JSON file in streaming mode (see print_json_file_data)
  # rag.print_json_file_data(limit=10)

  # Init search in terminal mode
  # rag.init_search_terminal_mode(n_results=10)
