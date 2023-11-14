#### FastAPI框架

- [官方示例](https://fastapi.tiangolo.com/advanced/testing-websockets/)

- 服务端（后端）代码

  ```python
  # backend/app/main.py
  import asyncio
  import logging
  from datetime import datetime
  
  from fastapi import FastAPI, WebSocket, WebSocketDisconnect
  
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger("FastAPI app")
  
  app = FastAPI()
  
  
  async def heavy_data_processing(data: dict):
      """Some (fake) heavy data processing logic."""
      await asyncio.sleep(2)
      message_processed = data.get("message", "").upper()
      return message_processed
  
  
  # Note that the verb is `websocket` here, not `get`, `post`, etc.
  @app.websocket("/ws")
  async def websocket_endpoint(websocket: WebSocket):
      # Accept the connection from a client.
      await websocket.accept()
  
      while True:
          try:
              # Receive the JSON data sent by a client.
              data = await websocket.receive_json()
              # Some (fake) heavey data processing logic.
              message_processed = await heavy_data_processing(data)
              # Send JSON data to the client.
              await websocket.send_json(
                  {
                      "message": message_processed,
                      "time": datetime.now().strftime("%H:%M:%S"),
                  }
              )
          except WebSocketDisconnect:
              logger.info("The connection is closed.")
              break
  ```

- 示例2:[代码出处](https://plainenglish.io/blog/websockets-in-python-fastapi-fetching-data-at-super-speed-3c7b355949fd#http-requests-in-python-fastapi)

  ```python
  import uvicorn
  from fastapi import FastAPI, WebSocket
  
  app = FastAPI()
  
  @app.websocket("/test")
  async def test(websocket: WebSocket):
      await websocket.accept()
      while True:
          request = await websocket.receive_json()
          message = request["message"]
          for i in range(10000):
              #  await websocket.send_text(str(i+1))
              await websocket.send_json({
                  "message": f"{message} - {i+1}",
                  "number": i+1
              })
  if __name__ == "__main__":
      uvicorn.run("app:app")
  ```

  

- 客户端代码

  - websockets库

  ```python
  # pip install websockets
  import asyncio
  from websockets.sync.client import connect
  
  def hello():
      with connect("ws://localhost:8765") as websocket:
          websocket.send("Hello world!")
          message = websocket.recv()
          print(f"Received: {message}")
  
  hello()
  ```

  - websocket库

  ```python
  # pip install websocket-client
  # pip install websocket
  import websocket
  
  url = "ws://127.0.0.1:20001/ws"
  ws = websocket.create_connection(url)
  ws.send("Hello World")
  resp = ws.recv()
  ws.close()
  ```

  