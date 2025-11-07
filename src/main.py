import chromadb



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
  testChromaClient()
