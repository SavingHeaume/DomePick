from flask import Flask, render_template, jsonify
from flask_sockets import Sockets
import json
from collections import deque

app = Flask(__name__)
sockets = Sockets(app)

# In-memory storage for demo purposes
ad_queues = {
    f"AdBoard {i}": deque(maxlen=3) for i in range(1, 6)
}

@app.route('/')
def index():
    return render_template('index.html')

@sockets.route('/stream')
def stream(ws):
    while not ws.closed:
        # Receive JSON data from the edge
        message = ws.receive()
        if message:
            try:
                data = json.loads(message)
                count = data.get("count", 0)
                results = data.get("results", [])

                # Prepare detailed information and recommendations
                received_info = []
                for person in results:
                    gender = person.get("gender", "Unknown")
                    age_range = person.get("age", "Unknown")
                    recommended_ad = recommend_ad(age_range, gender)
                    received_info.append({"gender": gender, "age": age_range, "ad": recommended_ad})

                # Update queues
                for board in ad_queues.keys():
                    for info in received_info:
                        ad_queues[board].append(info["ad"])

                # Send updated data back to the frontend
                ws.send(json.dumps({
                    "count": count,
                    "received_info": received_info,
                    "queues": {board: list(queue) for board, queue in ad_queues.items()}
                }))
            except json.JSONDecodeError:
                ws.send(json.dumps({"error": "Invalid JSON received"}))

def recommend_ad(age_range, gender):
    """Recommend an ad based on detailed age ranges and gender."""
    if gender == 1:  # Male
        if age_range == "0-2":
            return "Baby Products Ad"
        elif age_range == "4-6":
            return "Toys Ad"
        elif age_range == "8-13":
            return "Kids Sports Ad"
        elif age_range == "15-20":
            return "Gaming Gear Ad"
        elif age_range == "25-32":
            return "Smart Home Devices Ad"
        elif age_range == "38-43":
            return "Car Advertisement"
        elif age_range == "48-53":
            return "Travel Package Ad"
        elif age_range == "60+":
            return "Retirement Insurance Ad"
    elif gender == 0:  # Female
        if age_range == "0-2":
            return "Mother Care Ad"
        elif age_range == "4-6":
            return "Early Education Ad"
        elif age_range == "8-13":
            return "Arts & Crafts Ad"
        elif age_range == "15-20":
            return "Fashion Ad"
        elif age_range == "25-32":
            return "Luxury Fitness Ad"
        elif age_range == "38-43":
            return "Health Products Ad"
        elif age_range == "48-53":
            return "Wellness Ad"
        elif age_range == "60+":
            return "Elderly Care Ad"
    return "General Interest Ad"



if __name__ == '__main__':

    app.run(debug=True)