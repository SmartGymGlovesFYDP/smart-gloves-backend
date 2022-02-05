from flask import Flask

app = Flask(__name__)

# Define Routes below


@app.route("/")
def init():
    return "pass in a valid route"


@app.route("/exerciseData")
def exerciseData():
    return {"intensity": ["Mad", "Bad", "Sad"]}


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port='8000')
