from app import application as app, util
from flask import render_template, request
from .predict import predict


@app.route('/')
def index():
    return render_template('index.html', title='Code Breaker', page='Computer Vision API')


@app.route('/test', methods=['GET'])
def testRoute():
    data = {
        'key': 'value'
    }
    return util.success_response(200, 'This is a test response.', data)


@app.route('/solve', methods=['POST'])
def solveSudoku():
    data = request.get_json() or {}
    if len(data) > 0:
        quiz = data['quiz']
        response = {
            'solution': predict(quiz)
        }
        return util.success_response(200, 'Puzzle solved and solution returned.', response)

    return util.error_response(400, 'Error receiving the puzzle.')
