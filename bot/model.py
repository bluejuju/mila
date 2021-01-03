from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import tensorflow as tf

# Read text file convert to lower case and create a list of lines.
lines = open('bot\\train.txt', 'r', encoding='utf-8').read().lower().split('\n')

# Inputs, outputs converstation

inputs, outputs = [], []

for k in range(len(lines) - 1):
    inp = '\t' + lines[k] + '\n'
    out = '\t' + lines[k + 1] + '\n'

    inputs.append(inp)
    outputs.append(out)

chars = set()

for (inp, out) in zip(inputs, outputs):
    for ch in inp:
        chars.add(ch)
    for ch in out:
        chars.add(ch)

# Write chars to file
fwrite = open('bot\chars.txt', 'w', encoding='utf-8')
fwrite.write(''.join(chars))
fwrite.close()


# Map char to index, index to char
chr2idx = {}
idx2chr = {}

for k, ch in enumerate(chars):
    chr2idx[ch] = k
    idx2chr[k] = ch


# Create the training arrays
maxlen = max([len(x) for x in inputs])
maxlen = max([len(x) for x in outputs])

num_examples = len(inputs) # Total number of examples
num_tokens = len(chars)


# One-hot encoding, takes more RAM than Sparse.

encoder_inputs = np.zeros((num_examples, maxlen, num_tokens))
decoder_inputs = np.zeros((num_examples, maxlen, num_tokens))
decoder_outputs = np.zeros((num_examples, maxlen, num_tokens))

for i, (inp, out) in enumerate(zip(inputs, outputs)):
    for k, ch in enumerate(inp):
        encoder_inputs[i, k, chr2idx[ch]] = 1.0
    for k, ch in enumerate(out):
        decoder_inputs[i, k, chr2idx[ch]] = 1.0
        if k > 0:
            decoder_outputs[i, k - 1, chr2idx[ch]] = 1.0



latent_dim = 256

# Encoder
encoder_input                 = Input(shape=(None, num_tokens))
encoder                       = LSTM(latent_dim, return_state=True)
encoder_out, state_h, state_c = encoder(encoder_input)
encoder_states = [state_h, state_c]

# Decoder
decoder_input     = Input(shape=(None, num_tokens))
decoder           = LSTM(latent_dim, return_state=True, return_sequences=True)
decoder_out, _, _ = decoder(decoder_input, initial_state=encoder_states)
fc                = Dense(num_tokens, activation='softmax')
decoder_output    = fc(decoder_out)

model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()


model.fit([encoder_inputs, decoder_inputs], decoder_outputs, epochs=2048)
model.save('bot\chatbot.h5')
