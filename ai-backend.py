from flask import Flask, request, render_template
app = Flask(__name__)
@app.route('/process', methods=['get'])
def process_input():
    # 'request.args' is used to get the query string parameters for GET requests
    user_input = request.args.get('user_input', '')  # Default to empty string if no input
    output = f"You entered: {user_input}"
    return output

if __name__ == '__main__':
    app.run(debug=True)
    
    