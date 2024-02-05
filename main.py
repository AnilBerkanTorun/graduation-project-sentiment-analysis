import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('hepsiburada.csv')
#print(dataset)

target = dataset['Rating'].values.tolist()
data = dataset['Review'].values.tolist()

cutoff = int(len(data) * 0.80)
x_train, x_test = data[:cutoff], data[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]

#print(x_train[500])
#print(x_train[800])
#print(y_train[800])

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data)
#print(tokenizer.word_index)

x_train_tokens = tokenizer.texts_to_sequences(x_train)
#print(x_train[800])
#print(x_train_tokens[800])

x_test_tokens = tokenizer.texts_to_sequences(x_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

#print(np.mean(num_tokens))
#print(np.max(num_tokens))
#print(np.argmax(num_tokens))

#print(x_train[21941])

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
#print(max_tokens)
#print(np.sum(num_tokens < max_tokens) / len(num_tokens))

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)
#print(x_train_pad.shape)
#print(x_test_pad.shape)
#print(np.array(x_train_tokens[800]))
#print(x_train_pad[800])

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token!=0]
    text = ' '.join(words)
    return text

#print(x_train[800])
#print(tokens_to_string(x_train_tokens[800]))

model = Sequential()
embedding_size = 50

model.add(Embedding(input_dim=num_words+1,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))

model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#print(model.summary())

y_train = np.array(y_train)
y_test = np.array(y_test)

# Modelin eğitimi
model.fit(x_train_pad, y_train,
          validation_split=0.1, epochs=8, batch_size=256)

# Modelin değerlendirilmesi
result = model.evaluate(x_test_pad, y_test)
#print("Test Loss:", result[0])
#print("Test Accuracy:", result[1])

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])

cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

#print(len(incorrect))

idx = incorrect[0]
#print(idx)

text = x_test[idx]
#print(text)

#print(y_pred[idx])

#print(cls_true[idx])

text1 = "Keşke bu ürünü almasaydım"
text2 = "Rengi  resimde görüldügü gibi geldi"
text3 = "Ürünü beğenmedim "
text4 = "bu yaptığınız dolandırıcılıktır"
text5 = "Sakın Almayın !"
text6 = "Hayran kaldım"
text7 = "Aldıgım ayakkabının derisi yırtık geldi"
text8 = "Gerçekten çok kaliteli ürünmüş"
text9 = "Asla tavsiye etmiyorum"
text10 = "tek kelimeyle şahane bir ceketmiş"
texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]

tokens = tokenizer.texts_to_sequences(texts)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
print(tokens_pad.shape)

print(model.predict(tokens_pad))