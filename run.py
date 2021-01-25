from flask import Flask, request
from src.utils import utils
app = Flask(__name__)

@app.route('/')
def home():
   return "hello world!"

@app.route('/invocations', methods=['POST'])
def giveRecommendations():
    data = request.get_json()
    user_id = data['user_id_hashes'][0]
    util = utils.utils(user_id)
    response = util.model()
    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8080)