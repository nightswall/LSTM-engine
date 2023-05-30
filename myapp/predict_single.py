from model import Model
import numpy as np
import json

flow_types = {0: "BruteForce", 1: "dos", 3: "legitimate", 4: "malformed", 2: "SlowITe", 5: "Flooding"}
model_checkpoint = "checkpoints/ckpt70_reduced/ckpt70_reduced.ckpt"
session_checkpoint = "checkpoints/ckpt70_reduced/session.ckpt"


def get_prediction(incoming_message = None):
	global flow_types, model_checkpoint, session_checkpoint
	model, _ = Model().create_model(33, model_checkpoint) # Creating a new model for reference
	model = model.load_model(model, model_checkpoint, session_checkpoint) # Loading model checkpoint and session

	print(incoming_message)

	if incoming_message is not None: # Checking if a valid request is made
		prediction = model.predict(incoming_message) # Prediction from the engine
		prediction = np.argmax(prediction, axis = 1)
		result = []

		for p in prediction:
			f = dict()
			if flow_types[p] == "legitimate":
				f = {"type": "LEGITIMATE", "prediction": flow_types[p]}
			else:
				f = {"type": "MALICIOUS", "prediction": flow_types[p]} 
			result.append(f)

		decisions = {"type": "RESPONSE", "predictions": result} # Decision
		return response
