# modeling.py
def train_and_save_model(csv_path):
    print("Mulai training...")

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    import pickle
    import os

    df = pd.read_csv(csv_path)
    X = df['comment'].astype(str)
    y = df['sentimen']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

    tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    with open("model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

    model = Sequential([
        Embedding(1000, 128, input_length=100),
        LSTM(128),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_seq, y_train, epochs=5, validation_data=(X_test_seq, y_test))

    if not os.path.exists("model"):
        os.makedirs("model")
    model.save("model/sentiment_model.h5")
    print("Model dan tokenizer berhasil disimpan!")

# Tambahkan ini agar file bisa dijalankan langsung
if __name__ == "__main__":
    train_and_save_model("data/data_youtube_ijazah_jokowi.csv")
