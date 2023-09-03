# Text Sequence Generator using NLP

## Introduction

A Text Sequence Generator using NLP is a model or system designed to generate coherent and contextually relevant text sequences. These generators utilize Natural Language Processing (NLP) techniques and often rely on deep learning models, such as recurrent neural networks (RNNs). 

## Text Preprocessing:

You start with a large corpus of text data. This corpus can be anything from books, articles, social media posts, or any other text source relevant to your task.

**Corpus of text**

> "The movie was absolutely fantastic! The plot was engaging, and the acting was superb. I was on the edge of my seat the entire time.\n I couldn't stand this movie. The story was confusing, and the characters felt flat. I regret watching it.\nThis film had its moments, but overall, it failed to deliver. The pacing was off, and the dialogue felt forced.\nI'm in awe of the cinematography in this movie. The visual effects were stunning, and it added a whole new dimension to the story.\nThe acting was mediocre at best, and I found myself checking my watch multiple times during the film. A disappointment.\nI laughed so hard during this comedy. The humor was fresh, and the chemistry between the actors was palpable.\nA true masterpiece. The direction, screenplay, and performances all came together to create an unforgettable cinematic experience.\nThe horror elements in this movie were spot on. I was genuinely scared, and the suspense was maintained throughout.\nThe romantic scenes felt forced and unrealistic. It's a shame because the premise of the movie had potential.\nBoring and unoriginal. I've seen this plot a hundred times before, and this movie didn't bring anything new to the table."

You preprocess the text, which typically involves tokenization (splitting the text into words or subword units), lowercasing, and possibly removing punctuation and special characters. This can be achieved using the tokenizer class from the keras library.

The tokenizer class tokenizes a corpus of text, converts it to lowercase, builds a vocabulary of unique words, and assigns integer indices to each word. This tokenization process is a common initial step in many NLP tasks, such as text classification, sentiment analysis, and language modeling.

***Vocabulary of unique words for the above corpus of text.***

    {'the': 1, 'was': 2, 'and': 3, 'this': 4, 'movie': 5, 'i': 6, 'a': 7, 'to': 8, 'of': 9, 'felt': 10, 'it': 11, 'in': 12, 'plot': 13, 'acting': 14, 'on': 15, 'my': 16, 'story': 17, 'film': 18, 'had': 19, 'forced': 20, 'were': 21, 'new': 22, 'times': 23, 'during': 24, 'absolutely': 25, 'fantastic': 26, 'engaging': 27, 'superb': 28, 'edge': 29, 'seat': 30, 'entire': 31, 'time': 32, "couldn't": 33, 'stand': 34, 'confusing': 35, 'characters': 36, 'flat': 37, 'regret': 38, 'watching': 39, 'its': 40, 'moments': 41, 'but': 42, 'overall': 43, 'failed': 44, 'deliver': 45, 'pacing': 46, 'off': 47, 'dialogue': 48, "i'm": 49, 'awe': 50, 'cinematography': 51, 'visual': 52, 'effects': 53, 'stunning': 54, 'added': 55, 'whole': 56, 'dimension': 57, 'mediocre': 58, 'at': 59, 'best': 60, 'found': 61, 'myself': 62, 'checking': 63, 'watch': 64, 'multiple': 65, 'disappointment': 66, 'laughed': 67, 'so': 68, 'hard': 69, 'comedy': 70, 'humor': 71, 'fresh': 72, 'chemistry': 73, 'between': 74, 'actors': 75, 'palpable': 76, 'true': 77, 'masterpiece': 78, 'direction': 79, 'screenplay': 80, 'performances': 81, 'all': 82, 'came': 83, 'together': 84, 'create': 85, 'an': 86, 'unforgettable': 87, 'cinematic': 88, 'experience': 89, 'horror': 90, 'elements': 91, 'spot': 92, 'genuinely': 93, 'scared': 94, 'suspense': 95, 'maintained': 96, 'throughout': 97, 'romantic': 98, 'scenes': 99, 'unrealistic': 100, "it's": 101, 'shame': 102, 'because': 103, 'premise': 104, 'potential': 105, 'boring': 106, 'unoriginal': 107, "i've": 108, 'seen': 109, 'hundred': 110, 'before': 111, "didn't": 112, 'bring': 113, 'anything': 114, 'table': 115}
    
It is the word-to-index mapping that the tokenizer has learned. It will display a dictionary where words are keys, and their corresponding integer indices are values.


**Input Sequences Generation**

-   It initializes an empty list to store input-output pairs for training the text generation model.
-   It iterates through each line (text segment) in the `corpus`.
-   For each line, it converts the text into a sequence of integer values using a pre-trained `tokenizer`. This mapping converts each word in the text into an integer based on the tokenizer's vocabulary.



**Some of the generated sequences from the texts** 

> the movie was absolutely fantastic! the plot was engaging, and the acting was superb. i was on the edge of my seat the entire time.
[1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9, 16, 30, 1, 31, 32]

 >i couldn't stand this movie. the story was confusing, and the characters felt flat. i regret watching it.
[6, 33, 34, 4, 5, 1, 17, 2, 35, 3, 1, 36, 10, 37, 6, 38, 39, 11]


- Then, it generates multiple input-output pairs by considering all possible sub-sequences of the original sequence.
> the movie was absolutely fantastic! the plot was engaging, and the acting was superb. i was on the edge of my seat the entire time.

    [[1, 5], [1, 5, 2], [1, 5, 2, 25], [1, 5, 2, 25, 26], [1, 5, 2, 25, 26, 1], [1, 5, 2, 25, 26, 1, 13], [1, 5, 2, 25, 26, 1, 13, 2], [1, 5, 2, 25, 26, 1, 13, 2, 27], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9, 16], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9, 16, 30], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9, 16, 30, 1], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9, 16, 30, 1, 31], [1, 5, 2, 25, 26, 1, 13, 2, 27, 3, 1, 14, 2, 28, 6, 2, 15, 1, 29, 9, 16, 30, 1, 31, 32]]

These sub-sequences are used as input sequences, and the last element of each sub-sequence is used as the corresponding label or output.

**Padding Sequences:**

Padding is applied to sequences to make them all have the same length. Padding is added to the beginning of each sequence ('pre' padding) using the `pad_sequences` function.

    [[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
	   5]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5
	   2]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2
	  25]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25
	  26]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26
	   1]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1
	  13]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13
	   2]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2
	  27]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27
	   3]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3
	   1]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1
	  14]
	 [ 0  0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14
	   2]
	 [ 0  0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2
	  28]
	 [ 0  0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28
	   6]
	 [ 0  0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6
	   2]
	 [ 0  0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2
	  15]
	 [ 0  0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15
	   1]
	 [ 0  0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1
	  29]
	 [ 0  0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1 29
	   9]
	 [ 0  0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1 29  9
	  16]
	 [ 0  0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1 29  9 16
	  30]
	 [ 0  0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1 29  9 16 30
	   1]
	 [ 0  1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1 29  9 16 30  1
	  31]
	 [ 1  5  2 25 26  1 13  2 27  3  1 14  2 28  6  2 15  1 29  9 16 30  1 31
	  32]]

**Splitting Data:**

-   `xs`: These are the input sequences, containing all elements of each sequence except the last one.
-   `labels`: These are the labels or target values, representing the last element of each input sequence.


**One-Hot Encoding:**

We perform one-hot encoding on the `labels` to convert them into categorical vectors. The number of categories is determined by the total number of unique words in the tokenizer's vocabulary.

The resulting `xs` and `ys` are used as training data for a text generation model. `xs` are the input sequences, `ys` are the corresponding one-hot encoded labels, and the model is trained to predict the next word in a sequence given the previous words. This is a common data preparation process for sequence-to-sequence models in NLP tasks like text generation and language modeling.

## Training the model

The model consists of an Embedding layer to convert input sequences into dense vectors, a Bidirectional LSTM layer to capture contextual information bidirectionally, and a Dense output layer with a softmax activation to predict the next word in a sequence. This architecture is suitable for text sequence generation tasks, where the model learns to generate coherent and contextually relevant text based on the input context.

    model = Sequential([
	    Embedding(total_words, 64, input_length = max_sequence_len - 1),
	    Bidirectional(LSTM(20)),
	    Dense(total_words, activation = 'softmax')
	])

	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	history = model.fit(xs, ys, epochs = 500, verbose = 1)

The model is trained for 500 epochs.

> Accuracy of the model

![enter image description here](https://github.com/Satzil/Text-sequence-generator/blob/main/images/accuracy.png?raw=true)

The model has shown stable accuracy after the 250th epoch. The accuracy was approximated to be almost 95% at the end of the 500th epoch.

## Text Generation

The process starts with an initial seed text, which serves as the starting point for text generation. In this case, the seed text is "this movie was."

The variable `next_words` determines how many additional words or tokens of text should be generated after the seed text. In this example, it's set to generate 100 words.

    seed_text = "this movie was"
	next_words = 100

The code enters a loop that iterates for `next_words` times. In each iteration, it generates one word at a time to extend the text.

To ensure that the tokenized sequence has the same length as expected by the model, padding is applied. Padding adds zeros to the beginning of the sequence to match the model's input length.

**Updating Seed Text**

The predicted word is added to the `seed_text` to extend the context for the next iteration of the loop. This updated `seed_text` now includes the previously generated words.

**Printing the Generated Text**

After all iterations are completed, the code prints the generated text. This text includes the initial seed text and the additional words generated during the loop.

***Output***

> this movie was absolutely fantastic the plot was engaging and the acting was superb i was on the edge of my seat the entire time time time time entire time time time cinematic anything bring anything new to the story the table table the flat cinematic unforgettable palpable palpable palpable palpable cinematic the cinematic unforgettable palpable potential potential cinematic the potential cinematic the entire time time potential potential entire time time time time cinematic the cinematic the disappointment cinematic the potential entire time time time cinematic the potential entire time time time time entire time time time cinematic the cinematic the potential cinematic

The generated text appears to have repetitions and lacks coherence, which is a common issue in text generation models. This can happen due to various reasons, including limitations in the training data, model architecture, and training process.

We have to ensure that the training data is diverse and high-quality. The model's output is often influenced by the patterns and quality of the data it was trained on. A more extensive and diverse dataset can lead to better text generation.





