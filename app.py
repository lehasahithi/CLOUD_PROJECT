from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>My Flask App</title>
      </head>
      <body>
        <h1>Welcome to My Flask App!</h1>
        <p>This is a simple webpage served from Flask.</p>
        <img src="C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\sem_05\\CC\\docker(1).jpg" alt="Sample Image">
      </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
