# shakespeare <!-- omit in toc -->
## goal of this project
Write a neural network that can generate text that is similar in style to the
works of famous playwright William Shakespeare. For example, the following
excerpt from Act 1 of *Hamlet*:


>BERNARDO  
>I have seen nothing.
>
>
>MARCELLUS  \
>Horatio says 'tis but our fantasy, \
>And will not let belief take hold of him \
>Touching this dreaded sight, twice seen of us: \
>Therefore I have entreated him along \
>With us to watch the minutes of this night; \
>That if again this apparition come, \
>He may approve our eyes and speak to it. 
>
>HORATIO  \
>Tush, tush, 'twill not appear.
>

## results
Overall, I think the neural network can produce some pretty good results. It imitates Shakespeare's styles pretty well for a beginner project like this. Below are some examples I cherry picked. You can find more examples in `output.txt`, or you can try to run the code yourself. After training, use the method `generate_sample` function to generate text.

Regarding the below examples, let me make *make some excuses for why the results are not quite up to Shakespeare's standards*:
1. The network was only trained for 15 minutes, although I am not sure it would get that much better with more training.
2. I used character embedding, not word embedding, which accounts for all the typos. It is quite hard for a neural network to look at a big string of characters and infer how to spell words.
3. `RNN` cells (Reccurent Neural Networks) have the problem that as the distance between two particular characters increases, gradients tend to explode or vanish, meaning that information from many characters away is pretty much gone. This means the neural network is not able to learn to produce coherent text with more than a few characters. `LSTM` cells (Long Short Term Memory) attempt to improve on this, using a *gating mechanism* to allow the neural network to retain information for longer. However, the problem still persists, albeit in a smaller degree. This is neither the time nor the place to go into details on the inner workings of an `LSTM` cell, but see [Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory) for more details.

### example 1: Gonzalo loses his ability to speak
>ANTONIO:<br>
>I should stand, and again? Strife! Cursed my heart:<br>
>I'll all devil's art my brother in his daughter<br>
>in the blown; unstreatishs: sigieud; lo oath!<br>
><br>
>GONZALO:<br>
>What of you; sees all in that in the subject,<br>
>Would not well not both haal't her.<br>
><br>
>SEBASTIAN:<br>
>Good mistress art more.<br>
><br>
>GONZALO:<br>
><br>
>PROSPERO:<br>
>A brother, the swain make, for accorded<br>
>And at thy blessens, to then such us of sick<br>
>With the more and dropp.<br>
>One of Bianca in Lancal. Come, look!<br>

### example 2: *stabbiticulation*
>ARIEL:<br>
>What of the dukedom, and the business do unwo:<br>
>And here's gone?<br>
><br>
>ARIEL:<br>
>Here without your mis, sir, if they will say none.<br>
><br>
>SEBASTIAN:<br>
>Not thank, all rail it.<br>
><br>
>SEBASTIAN:<br>
>My name.'<br>
><br>
>GONZALO:<br>
>Would, my leeving, or weep your drunking.<br>
><br>
>SEBASTIAN:<br>
>By which would I say;<br>
>Awake you alone,--a knocked wash; you did,<br>
>Which in our father,<br>
>First'd heaven lord my special. Sir, let's not be his<br>
>sovere in each senge.<br>
><br>
>ANTONIO:<br>
>He's into yourself: he shall make his droppes before!<br>
><br>
>TRAN:<br>
>Stabbiticulation.<br>

Obviously the text generated is all nonsense; most of the time the neural network doesn't spell out actual words, and when it does, the phrases/sentences it comes up with make no sense. But I didn't expect that coming into this either. This was a pretty simple neural network with only one `LSTM` layer and one `Linear` layer with hidden size `250`, so we can't expect that much. I also only trained it for `10` epochs, which took about `15` minutes on the GPUs at Google Colab
## training data
big text file of shakespeare passages: [shakespeare.txt](shakespeare.txt), in particular about 40 000 lines

## input data
a string of characters that the neural network should base its text sample on. We should be able to give it even just a one character string, and it should be able to generate a long string of text.

## output
variable length sample of text that is supposed to be similar in style to Shakespeare's plays. The neural network will train on `shakespeare.txt`, and then we will define a function 
```
def generate_sample(length: int):
    # return a sample of the correct length in Shakespeare's style
```
which we can then call to generate a sample of any length we want
this is done in the following way:
1. Given the one character input, predict the next character
2. Use the character we just predicted to predict another character
3. Use the character predicted in step 2 to predict another character
4. Repeat until we have as many characters as we want



## model
`LSTM` model using **character embedding**<br>
The neural network looks at one character, then tries to predict the next character.
```
lstm = torch.nn.LSTM(input_size=65,
              hidden_size=250,
              num_layers=1
)
lstm --> torch.nn.Linear(250, 65) --> output shape (100, 1, 65)
targets shape (100, 1, 1)
 output, targets --> CrossEntropyLoss() --> loss shape (100, 1)
```
the neural network looks at `100` characters and predicts the next character for each of those characters. When we want to generate text we use `(1, 1, 65)` inputs instead and get `(1, 1, 65)` outputs accordingly.
The optimizer used is `Adam` (`torch.optim.Adam`), which, afaik is a fancy version of `SGD` (Stochastic Gradient Descent), although I am not sure about the details. I just have a vague understanding that it is 'better' somehow.


### model hyperparameters
* `num_layers = 1`, the number of `LSTM` layers
* `hidden_size = 250`, the hidden size of the `Linear` layer
* `seq_length = 100`, the number of characters to predict at once
* `lr = 0.01`, the learning rate of the model 
* `n_epochs = 10`, the number of epochs to train for
* `vocab_size = 65`, the number of unique characters in the input file. Not strictly a hyperparameter, since it is determined by the input file given

