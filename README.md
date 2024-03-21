# Digit Classifier

This project is a digit classifier implemented in PyTorch. It uses the MNIST dataset to train a neural network to recognize handwritten digits.

## Structure

The project is organized as follows:

- `DataLoader.py`: Contains code for loading the MNIST dataset.
- `Digit_classifier.ipynb`: A Jupyter notebook that trains and evaluates the Machine Learning model.
- `Digit_classifier.pth`: The trained model weights.
- `GUI.py`: A graphical user interface for using the digit classifier.
- `data/MNIST/`: The MNIST dataset.

## Usage

To use this project, first install the required dependencies:

```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
pip install -r requirements.txt
```

You can then run the GUI.py file

```sh
python GUI.py
```

## Contributing
Contributions are welcome. 
Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
