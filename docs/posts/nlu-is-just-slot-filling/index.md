<!--
.. title: NLU is just slot filling
.. slug: nlu-is-just-slot-filling
.. date: 2020-10-12 22:08:11 UTC+02:00
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text
-->


# Intro

This post is a compilation of my ideas about how NLU should look like in a task-oriented dialog system.
Any statement without a link is my opinion.


## tl;dr

- [NLU](https://en.wikipedia.org/wiki/Natural-language_understanding) is actually just slot filling
- use [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) + grammars and a whole text classifier

<!-- TEASER_END -->

# Context

Let's say you're building a [dialog system](https://en.wikipedia.org/wiki/Dialogue_system) to allow customers of some company to get things done without talking to a real human in a call center.
Chances are you're going to use a [task-oriented dialog system](https://arxiv.org/pdf/2003.07490.pdf) framework to build your software. Its architecture looks like a tuning fork:

![](/images/dialog_system_tuning_fork.svg)

There are many ways to implement components of such system, e.g. a dialog manager can be [rule-based](https://web.stanford.edu/~jurafsky/slp3/24.pdf), [graph-based](https://lekta.ai/), or a [supervised ML model](https://rasa.com/docs/rasa/stories). 
The component we'll focus on for the sake of this post is NLU; usually, it's a subsystem bounded by having user utterance text as an input (sometimes with additional dialog context information) and producing a [semantic frame](https://hao-fang.github.io/ee596_spr2018/slides/week_2-spoken_language_understanding.pdf), structured representation of a user utterance usable by dialog manager to update the dialog state and drive the further conversation.


# Slots and entities 

What data exactly and in what shape is contained in a semantic frame differs between the implementations, but it can be generalized to a collection of key-value pairs that I'll call _slot fills_. 
An example semantic frame for a first utterance in a flight booking assistant application could look like this:

```python
nlu("yo, tell me what flights to Berlin are available next week")
->
[["courtesy", "greeting"],
 ["intent", "search_flights"],
 ["flight_destination_city", "BER"],
 ["flight_window_start", "2020-10-12"],
 ["flight_window_end", "2020-10-19"]]
```

A _slot_ can be thought of as an atomic information container. 
A _dialog task_ (AKA _dialog strategy_, _dialog skill_) defines the bot's behavior by reacting to slot state changes and requesting slot values from the user in a form of an explicit prompt, or an external system e.g. in a form of HTTP requests.

Slots can be divided into two groups depending on the nature of the data that they accept:

1. Non-categorical slots. 
   You need to parse the underlying text to extract the value in a structured format, so you do care where exactly in the utterance this information occurs.
   Usually, it's difficult to enumerate each possible value a non-categorical slot can take.
   Non-categorical slots accept time, date, number, phone number, credit card number, etc.
   In the above snippet, `flight_destination`, `flight_window_start`, and `flight_window_end` are examples of this.
   The task of detecting these is [traditionally called](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU_final_doubleColumn.pdf) slot filling, but IMHO it's not enough.
2. Categorical slots. 
   These are slots that accept a value from a well-defined set of possible cases (think of an enum).
   You only care _what_ value occurred, not _where_ it occurred in the utterance.
   In the above snippet, `courtesy` and `intent` are examples of this.

Non-categorical slots are similar to _entities_ in the [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) problem.
The difference is that NER usually concerns universally accepted entities, like _person_, _date_, or _place_. 
Slots are project-dependent and a single dialog system can contain multiple slots that accept similar, but distinct values, like _flight window start_ and _flight window end_, or _flight departure_ and _flight destination_.
Another variation is that NER returns merely the location of an entity in the utterance and a slot needs a structured value, not a free-form utterance substring.

While conceptually different, we'll use tools that train NER models to train a token tagger model to recognize project-specific slot locations.

# Slot filling model

The overall pipeline looks like this.

![](/images/slot_filling_model_overall.svg)

## Span Recognizer

It's a box that detects what slots are mentioned in the text and returns the substring positions.
Assuming a standard word tokenization, an example slot recognition can look like this:

```python
# word index     0   1    2  3    4       5  6      7   8         9    10
span_recognizer("yo, tell me what flights to Berlin are available next week")
# spans                                      <-A-->               <---B--->
#                                                                 <---C--->
->
[["flight_destination_city", [6, 1]],  # span A
 ["flight_window_start", [9, 2]], # span B
 ["flight_window_end", [9, 2]]]   # span C
```

If you ever [played with a NER model](https://demo.allennlp.org/named-entity-recognition/MjMyNjk4Mg==) this should look familiar.

We'll build Span Recognizer as a supervised token classifier with following approaches:

- Conditional Random Field &ndash; the way to do NER in classical ML

- fastText embeddings + LSTM &ndash; transfer learning to utilize pretrained knowledge about the world + an RNN to look at the token context in the utterance

- BERT or another transformer architecture (TBD)

See next posts for details.

## Value Extractor

Value Extractor extracts structured value from utterance substring returned by Span Recognizer.

Continuing the example about flight booking:

```python
value_extractor("yo, tell me what flights to Berlin are available next week", "flight_destination_city", [6, 1])
->
["flight_destination_city", "BER"]

value_extractor("yo, tell me what flights to Berlin are available next week", "flight_window_start", [9, 2])
->
["flight_window_start", "2020-10-12"]

value_extractor("yo, tell me what flights to Berlin are available next week", "flight_window_end", [9, 2])
->
["flight_window_end", "2020-10-19"]
```


Another example in a top up dialog could look like this:

```python
#                0    1  2 3   4  5   6  7
value_extractor("make me a top up for 20 bucks", "top_up_value", [6, 2])
->
["top_up_value", "20"]

#                0    1  2 3   4  5   6  7
value_extractor("make me a top up for 20 bucks", "top_up_currency", [6, 2])
->
["top_up_currency", "USD"]
```

Like other black boxes in our diagram, Value Extractor is just an _interface_. The actual implementation is whatever makes sense for a given project, e.g.:

- regex + custom code
- [rule-based](https://duckling.wit.ai/)
- [Geonames API](http://www.geonames.org/search.html?q=berlin&country=)


## Classifier

It's a simple [multi label text classifier](https://medium.com/@MageshDominator/machine-learning-based-multi-label-text-classification-9a0e17f88bb4) with a label mangling post processing.
Text goes in, a list of predicted classes comes out.


```python
classifier("yo, tell me what flights to Berlin are available next week")
->
[["courtesy", "greeting"],
 ["intent", "search_flights"]]

classifier("make me a top up for 20 bucks")
->
[["intent", "top_up"]]
```

The _mangling_ can be needed because label encoders in many ML frameworks support labels in a form of simple strings.

Let's say your dialog system consists of a couple of intents and categorical values. 
All slots values can be enumerated:

- `["intent", "top_up"]`
- `["intent", "search_flights"]`
- `["courtesy", "greeting"]`
- `["courtesy", "thank_you"]`
- `["travel_class", "business"]`
- `["travel_class", "premium_economy"]`
- `["travel_class", "economy"]`

It can be mangled, so that the classifier's label encoder only sees single strings:

- `"intent/top_up"`
- `"intent/search_flights"`
- `"courtesy/greeting"`
- `"courtesy/thank_you"`
- `"travel_class/business"`
- `"travel_class/premium_economy"`
- `"travel_class/economy"`

Note that this is possible for data like intents or categorical variables because all values can be enumerated easily.
It wouldn't be feasible to enumerate all 32-bit integers, even though it's theoretically possible:

- `"top_up_value/0"`
- `"top_up_value/1"`
- `"top_up_value/2"`
- ...
- `"top_up_value/4294967295"`

