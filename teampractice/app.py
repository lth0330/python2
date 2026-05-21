from fastapi import FastAPI
import uvicorn

app = FastAPI() 

if __name__ == "__main__" :
    uvicorn.run( 'app:app' , host='0.0.0.0' , port = 8000 , reload=True )  # 서버 수정 

import controller
app.include_router( controller.router )
