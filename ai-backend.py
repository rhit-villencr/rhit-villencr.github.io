from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process_input():
    # Get query parameters from the GET request
    user_input = request.args.get('user_input', '')  # Default to empty string if no input provided
    output = {"message": f"You entered: {user_input}"}
    
    # Return the result as a JSON response
    return jsonify(output)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
