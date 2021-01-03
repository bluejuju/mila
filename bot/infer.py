from numpy.core.records import _deprecate_shape_0_as_None
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import numpy as np

model = load_model('bot\chatbot.h5')

# Load chars to list of chars
chars = list(open('bot\chars.txt', 'r', encoding='utf-8').read())

# Map char to index, index to char
chr2idx = {}
idx2chr = {}

for k, ch in enumerate(chars):
    chr2idx[ch] = k
    idx2chr[k] = ch


latent_dim = 256

encoder_input = model.input[0]
encoder_out, state_h, state_c = model.layers[2].output # Encoder
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_input, encoder_states)

decoder_input = model.input[1]
# Create two placeholder
decoder_input_h = Input(shape=(latent_dim,), name='decoder_h')
decoder_input_c = Input(shape=(latent_dim,), name='decoder_c')
decoder_input_states = [decoder_input_h, decoder_input_c]
decoder = model.layers[3]
decoder_out, decoder_h, decoder_c = decoder(decoder_input, initial_state=decoder_input_states)
decoder_states = [decoder_h, decoder_c]
fc      = model.layers[4]
decoder_output = fc(decoder_out)

decoder_model = Model([decoder_input] + decoder_input_states, [decoder_output] + decoder_states)


maxlen = 48

def get_output(text):
    text = text.lower()
    x = np.zeros((1, maxlen, len(chars)))
    for k, ch in enumerate(text):
        x[0, k, chr2idx[ch]] = 1.0

    states = encoder_model.predict(x) # Get the states

    seq = np.zeros((1, 1, len(chars)))
    seq[0, 0, chr2idx['\t']] = 1.0

    stop = False

    tokens = []

    while not stop:
        out, h, c = decoder_model.predict([seq] + states)
        idx = np.argmax(out[0, -1, :])
        ch = idx2chr[idx]
        tokens.append(ch)

        # Our condition
        if len(tokens) > maxlen or ch == '\n':
            stop = True

        seq = np.zeros((1, 1, len(chars)))
        seq[0, 0, idx] = 1.0
        states = [h, c]
    return ''.join(tokens)



while True:
    text = input('Enter something: ')
    print(get_output(text))