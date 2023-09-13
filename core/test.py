
import os
import numpy as np
import configuration
import importlib

classes = sorted("blues classical country disco hiphop jazz metal pop reggae rock".split(' '))
print("Using the order:", classes)

for fname in os.listdir():
    
    if not os.path.isdir(fname) or fname == "__pycache__":
        continue

    print(f"\nLoading '{fname}'")

    # Import module and load the model
    module = importlib.import_module(fname) 
    model = module.Model(configuration)
    model.load_model(f"./{fname}/{fname}.h5")

    n_classes = len(classes)
    predictions = np.zeros((n_classes, n_classes))

    # Get predictions
    for gi in range(n_classes):
        genre = classes[gi]
        genre_path = os.path.join(configuration.DATASET_AUDIO_ROOT, genre)
        for filename in os.listdir(genre_path):
            try:
                file_path = os.path.join(genre_path, filename)
                prediction = model.predict(file_path)
                pi = classes.index(prediction)
                # Add to prediction matrix
                predictions[pi][gi] += 1
            except:
                continue

    print("Prediction matrix:")
    print(predictions)

    cm = predictions # Confusion matrix

    # Process the results to get the final metrics

    e = 1e-8 # Epsilon to avoid divide by zero errors
    avg_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    for c in range(n_classes):
        TP = cm[c][c] + e
        FN = sum(cm[c]) - TP + e
        FP = sum(cm[:, c]) - TP + e
        TN = sum(sum(cm)) - TP - FN - FP + e
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
        # print(f"\nMetrics for class {c}:")
        # print("Accuracy:", accuracy)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1:", f1)
        avg_accuracy += accuracy
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1

    avg_accuracy /= n_classes
    avg_precision /= n_classes
    avg_recall /= n_classes
    avg_f1 /= n_classes

    print("\nModel metrics:")
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1)

    # Explicitly clean the memory allocated to GPU 
    del model 
