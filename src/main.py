'''
Copyright (c) 2025 Ibict Authors. All rights reserved.
'''

import chromadb

import ijson
from pathlib import Path # Path manipulation





class RagHandler:
  # region MARK: JSON parsing

  def get_json_file_info(self, json_filepath: str):
    '''
    Get information about a JSON file.

    Args:
      json_filepath (str): The path to the JSON file.

    Returns:
      tuple: A tuple containing the file name and its size in megabytes.
    '''

    path    = Path(json_filepath)
    mb_size = round(path.stat().st_size / (1024 ** 2), 2)
    return path, mb_size

  def parse_json_file(self, json_filepath: str, limit: int = None):
    '''
    Parse a JSON file in streaming mode (reads without loading entire file into memory).
    It also yields (produces) documents to the generator.

    Args:
      json_filepath (str): The path to the JSON file.
      limit (int, optional): The maximum number of objects to parse.
    '''

    parsed_amount = 0
    info          = self.get_json_file_info(json_filepath)

    print(f">> Starting to parse '{info[0].name}' ({info[1]:.2f} MB)")

    with open(json_filepath, 'rb') as file:
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
        docs = self.__parse_object(obj)

        # Deliver these documents to the caller (generator)
        for doc in docs:
          yield doc

        parsed_amount += 1

        # Display progress
        if parsed_amount % 1000 == 0:
          print(f"> {parsed_amount:,} objects parsed.")

        # Stop if limit is reached
        if limit and parsed_amount >= limit:
          break

  def __parse_object(self, obj: dict) -> tuple:
    '''
    Parse a single object from a JSON file.

    Args:
      obj (dict): A single object from a JSON file.

    Returns:
      tuple: A tuple of documents.
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

    return doc_1, doc_2, doc_3, doc_4, doc_5

  # endregion



  # region MARK: ChromaDB

  def create_client(self, path: str = 'vectordb'):
    '''
    Creates a ChromaDB client.

    Args:
      path (str, optional): The path to the ChromaDB database.
    '''

    # Create a ChromaDB persistent client
    self.client = chromadb.PersistentClient(path = path)

  def create_collection(self, name: str = 'data'):
    '''
    Creates a ChromaDB collection.

    Args:
      name (str, optional): The name of the collection.
    '''

    assert self.client, "Client is not created yet. Call create_client(...) first."

    # Create a collection
    self.collection = self.client.create_collection(name = name)

    # Display the number of objects in the collection
    print(f">> {self.collection.count()} objects found in the collection.")

  def search(self, query_text: str, n_results: int = 10):
    '''
    Search for relevant documents in the ChromaDB collection.
    It needs a vector database to be already created and filled with the embeddings of the documents.

    Args:
      query_text (str): The query text to search for.
      n_results (int, optional): The number of results to return.
    '''

    assert self.collection, "Collection is not created yet. Call create_collection(...) first."

    print(f"\n>> Requested query: \"{query_text}\"\n\n")

    # Search for relevant documents and their metadatas
    results = self.collection.query(query_texts = [query_text], n_results = n_results)

    # Get the metadatas and documents
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Display the results

    # Found results
    if documents:
      print(f">> Found {len(documents)} relevant results:\n")

      for i, (document, metadata) in enumerate(zip(documents, metadatas), 1):
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

  rag = RagHandler()

  # Get info about a JSON file
  # print(rag.get_json_file_info(json_filepath="../data/db.json"))

  # Parse a JSON file
  gen = rag.parse_json_file(json_filepath="../data/db.json", limit=10)
  for doc in gen:
    print(f">> Found new document for {doc.get('COD_CCN_PUBLICACAO')}")
    print(doc)

  # Init search in terminal mode
  # rag.init_search_terminal_mode(n_results=10)
