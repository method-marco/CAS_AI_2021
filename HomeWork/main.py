import homework.homework_starter as starter

if __name__ == '__main__':
    result_augmented = starter.start_sp500_binary_classification_with_augmented_data()
    result = starter.start_sp500_binary_classification()

    print(f"With Augmented Data:\nTest Accuracy: {(100 * result_augmented[0]):>5f}%, Avg test loss: {result_augmented[1]:>8f} \n")
    print(f"Original Data only:\nTest Accuracy: {(100 * result[0]):>5f}%, Avg test loss: {result[1]:>8f} \n")

