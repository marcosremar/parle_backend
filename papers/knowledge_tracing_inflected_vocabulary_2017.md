# **Knowledge Tracing in Sequential Learning of Inflected Vocabulary**

**Adithya Renduchintala** and **Philipp Koehn** and **Jason Eisner**
Department of Computer Science
Johns Hopkins University
_{_ adi.r,phi,eisner _}_ @jhu.edu



**Abstract**


We present a feature-rich knowledge
tracing method that captures a student’s
acquisition and retention of knowledge during a foreign language phrase learning task.
We model the student’s behavior as making
predictions under a log-linear model, and
adopt a neural gating mechanism to model
how the student updates their log-linear
parameters in response to feedback. The
gating mechanism allows the model to learn
complex patterns of retention and acquisition for each feature, while the log-linear
parameterization results in an interpretable
knowledge state. We collect human data
and evaluate several versions of the model.


**1** **Introduction**


_Knowledge tracing_ attempts to reconstruct when
a student acquired (or forgot) each of several
facts. Yet we often hear that “learning is not just
memorizing facts.” Facts are not atomic objects
to be discretely and independently manipulated.
Rather, we suppose, a student who recalls a fact in a
given setting is demonstrating a _skill_ —by solving a
structured prediction problem that is akin to reconstructive memory (Schacter, 1989; Posner, 1989)
or pattern completion (Hopfield, 1982; Smolensky,
1986). The attempt at structured prediction may
draw on many cooperating feature weights, some
of which may be shared with other facts or skills.
In this paper, for the task of foreign-language vocabularylearning, wewilladoptaspecificstructured
prediction model and learning algorithm. Different
knowledge states correspond to model parameter
settings (feature weights). Different learning styles
correspond to different hyperparameters that govern
the learning algorithm. [1] As we interact with each
student through a simple online tutoring system, we


1In the present paper, we assume that all students share the
same hyperparameters (same learning style), although each studentwillhavetheirownparameters, whichchangeastheylearn.



would like to track their evolving knowledge state
and identify their learning style. That is, we would
like to discover parameters and hyperparameters
that can explain the evidence so far and predict how
the student will react in future. This could help
us make good future choices about how to instruct
this student, although we leave this reinforcement
learning problem to future work. In this paper, we
show that we can predict the student’s next answer.

In short, we expand the notion of a knowledge
tracing model to include representations for a
student’s (i) current knowledge, (ii) retention of
knowledge, and (iii) acquisition of new knowledge.
Our reconstruction of the student’s knowledge
state remains interpretable, since it corresponds to
the weights of hand-designed features (sub-skills).
Interpretability may help a future teaching system
provide useful feedback to students and to human
teachers, and help it construct educational stimuli
that are targeted at improving particular sub-skills,
such as features that select correct verb suffixes.

Our present paper considers a verb conjugation
task, where a foreign language learner learns
the verb conjugation paradigm by reviewing and
interacting with a series of flash cards. This
task is a good testbed, as it needs the learner to
deploy sub-word features and to generalize to
new examples. For example, a student learning
Spanish verb conjugation might encounter pairs
such as (t´u entras, you enter), (yo miro,
I watch). Using these examples, the student
needs to recognize suffix patterns and apply them to
new pairs seen such as (yo entro, I enter).

Vocabulary learning presents a challenging
learning environment due to the large number of
skills (words) that need to be traced. Learning
vocabulary in conjunction with inflection further
complicates the challenge due to the number of new
sub-skills that are introduced. Huang et al. (2016)
suggest that modeling sub-skill interaction is crucial
to several knowledge tracing domains. For our
domain, a log-linear formulation elegantly allows
for arbitrary sub-skills via feature functions.



238


_Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)_, pages 238–247,
Vancouver, Canada, August 3 - August 4, 2017. c _⃝_ 2017 Association for Computational Linguistics


**2** **Related Work**


Bayesian knowledge tracing (Corbett and Anderson,
1994) (BKT) has long been the standard method to
infer a student’s knowledge from his or her performance on a sequence of task items. In BKT, each
skill is modeled by an HMM with two hidden states
(“known” or “not-known”), and the probability of
success on an item depends on the state of the skill it
exercises. Transition and emission probabilities are
learned from the performance data using Expectation Maximization (EM). Many extensions of BKT
have been investigated, including personalization
(e.g., Lee and Brunskill, 2012; Khajah et al., 2014a)
and modeling item difficulty (Khajah et al., 2014b).
Our approach could be called **Parametric**
**Knowledge Tracing (PKT)** because we take a
student’s knowledge to be a vector of prediction
parameters (feature weights) rather than a vector
of skill bits. Although several BKT variants
(Koedinger et al., 2011; Xu and Mostow, 2012;
Gonzalez-Brenes et al.´, 2014) have modeled the
fact that related skills share sub-skills or features,
that work does not associate a real-valued weight
with each feature at each time. Either skills are still

represented with separate HMMs, whose transition
and/or emission probabilities are parameterized in
termsofsharedfeatureswithtime-invariantweights;
or else HMMs are associated with the individual

sub-skills, and the performance of a skill depends
on which of its subskills are in the “known” state.

Our current version is not Bayesian since it
assumes deterministic updates (but see footnote 4).
A closely related line of work with deterministic
updates is deep knowledge tracing (DKT) (Piech
et al., 2015), which applied a classical LSTM model
(Hochreiter and Schmidhuber, 1997) to knowledge
tracing and showed strong improvements over
BKT. Our PKT model differs from DKT in that

the student’s state at each time step is a more
interpretable feature vector, and the state update rule
is also interpretable—it is a type of error-correcting
learning rule. In addition, the student’s state is
able to predict the student’s actual response and
not merely whether the response was correct.
We expect that having an interpretable feature
vector has better inductive bias (see experiment in
section 7.1), and that it may be useful to plan future
actions by smart flash card systems. Moreover, in
this work we test different plausible state update
rules and see how they fit actual student responses,
in orer to gain insight about learning.



Most recently, Settles and Meeder (2016)’s halflife regression assumes that a student’s retention
of a particular skill exponentially decays with time
and learns a parameter that models the rate of decay
(“half-life regression”). Like Gonzalez-Brenes et al.´
(2014) and Settles and Meeder (2016), our model
leverages a feature-rich formulation to predict the
probability of a learner correctly remembering a
skill, but can also capture complex spacing/retention
patterns using a neural gating mechanism. Another
distinction between our work and half-life regression is that we focus on knowledge tracing within
a single session, while half-life regression collapses
a session into a single data point and operates on
many such data points over longer time spans.


**3** **Verb Conjugation Task**


We devised a flash card training system to teach
verb conjugations in a foreign language. In this
study, we only asked the student to translate from
the foreign language to English, not vice-versa. [2]


**3.1** **Task Setup**


We consider a setting where students go through
a series of interactive flash cards during a training
session. Figure 1 shows the three types of cards:
(i) _Example_ (EX) cards simply display a foreign
phrase and its English translation (for 7 seconds).
(ii) _Multiple-Choice_ (MC) cards show a single
foreign phrase and require the student to select one
of five possible English phrases shown as options.
(iii) _Typing_ (TP) cards show a foreign phrase and a
text input box, requiring the student to type out what
they think is the English translation. g Our system
can provide feedback for each student response.
(i) _Indicative Feedback_ : This refers to marking a
student’s answer as correct or incorrect (Fig. 1c, 1d
and 1h). Indicative feedback is always shown for
both MC and TP cards. (ii) _Explicit Feedback_ : If
the student makes an error on a TP card, the system
has a 50% chance of showing them the true answer
(Fig. 1g). (iii) _Retry_ : If the student makes an error on
a MC card, the system has a 50% chance of allowing
them to try again, up to a maximum of 3 attempts.


**3.2** **Task Content**


In this particular task we used three verb lemmas,
each inflected in 13 different ways (Table 1). The
inflections included three tenses (simple past,


2We would regard these as two separate skills that share parameters to some degree, an interesting subject for future study.



239


(a) (b) (c) (d)


(e) (f) (g) (h)
Figure 1: Screen grabs of card modalities during training. These examples show cards for a native English speaker learning
Spanish verb conjugation. Fig 1a is an EX card, Fig 1b shows a MC card before the student has made a selection, and Fig 1c and 1d
show MC cards after the student has made an incorrect or correct selection respectively, Fig 1e shows a MC card that is giving
the student another attempt (the system randomly decides to give the student up to three additional attempts), Fig 1f shows a TP
card where a student is completing an answer, Fig 1g shows a TP card that has marked a student answer wrong and then revealed
the right answer (the reveal is decided randomly), and finally Fig 1h shows a card that is giving a student feedback for their answer.


Categories Inf SPre,1,N SPre,2,N SPre,3,M SPre,3,F SF,1,N SF,2,N SF,3,M SF,3,F SP,1,N SP,2,N SP,3,M SP,3,F


acceptar yo acepto t´u aceptas ´el acepta ella acepta yo aceptar´e t´u aceptar´as ´el aceptar´a ella aceptar´a yo acept´e t´u aceptaste ´el acept´o ella acept´o
to accept I accept you accept he accepts she accepts I will accept you will accept* he will accept she will accept I accepted* you accepted he accepted she accepted


Lemma entrar yo entro t´u entras ´el entra ella entra yo entrar´e t´u entrar´as ´el entrar´a ella entrar´a yo entr´e t´u entraste ´el entr´o ella entr´o
to enter I enter you enter he enters she enters I will enter you will enter he will enter she will enter I entered you entered he entered she entered


mirar yo miro t´u miras ´el mira ella mira yo mirar´e t´u mirar´as ´el mirar´a ella mirar´a yo mir´e t´u miraste ´el mir´o ella mir´o
to watch I watch* you watch* he watches* she watches I will watch you will watch* he will watch she will watch I watched you watched he watched* she watched


Table 1: Content used in training sequences. Phrase pairs with * were used for the quiz at the end of the training sequence. This
Spanish content was then transformed using the method in section 6.1.



present, and future) in each of four persons (first,
second, third masculine, third feminine), as well
as the infinitive form. We ensured that each surface
realization was unique and regular, resulting in 39
possible phrases. [3] Seven phrases from this set were
randomly selected for a quiz, which is shown at
the end of the training session, leaving 32 phrases
that a student may see in the training session. The
student’s responses on the quiz do not receive
any feedback from the system.We also limited the
training session to 35 cards (some of which may
require multiple rounds of interaction, owing to
retries). All of the methods presented in this paper
could be applied to larger content sets as well.


**4** **Notation**


We will use the following conventions in this paper.
System actions _at_, student responses _yt_, and feedback items _a_ _[′]_ _t_ [are subscripted by a time][ 1] _[ ≤]_ _[t][ ≤]_ _[T]_ [.]
Other subscripts pick out elements of vectors or matrices. Ordinary lowercase letters indicate scalars


3The inflected surface forms included explicit pronouns.



( _α_, _β_, etc.), boldfaced lowercase letters indicate
vectors ( _**θ**_, **y**, **w** [zx] ), and boldfaced uppercase letters
indicate matrices ( **Φ**, **W** [hh], etc.). The roman-font
superscripts are part of the vector or matrix name.


**5** **Student Models**


**5.1** **Observable Student Behavior**


A flash card is a structured object _a_ =( _x,O_ ), where
_x ∈X_ is the foreign phrase and _O_ is a set of allowed responses. For an MC card, _O_ is the set of 5
multiple-choice options on that card (or fewer on a
retry attempt). For a EX or TP card, _O_ is the set of all
39 English phrases (the TP user interface prevents
the student from submitting a guess outside this set).
For non-EX cards, we assume the student samples
their response _y ∈O_ from a log-linear distribution
parameterized by their knowledge state _**θ**_ _∈_ R _[d]_ :


_p_ ( _y_ _|_ _a_ ; _**θ**_ )= _p_ ( _y_ _|_ _x,O_ ; _**θ**_ )

exp( _**θ**_ _·_ _**φ**_ ( _x,y_ ))
=
(1)
~~�~~ _y_ _[′]_ _∈O_ [exp(] _**[θ]**_ _[·]_ _**[φ]**_ [(] _[x,y][′]_ [))]



240


where _**φ**_ ( _x,y_ ) _∈_ R _[d]_ is a feature vector extracted
from the ( _x,y_ ) pair.


**5.2** **Feature Design**


The student’s knowledge state is described by the
weights _**θ**_ placed on the features _**φ**_ ( _x,y_ ) in equation (1). We assume the following binary features
will suffice to describe the student’s behavior.

_• Phrasal_ features: We include a unique indicator
feature for each possible ( _x,y_ ) pair, yielding 39 [2]

features. For example, there exists a feature that
fires iff _x_ = yo miro _∧_ _y_ = I ~~e~~ nter.

_• Word_ features: We include indicator features for

all(sourceword, targetword)pairs: e.g., yo _∈_ _x_ _∧_
enter _∈_ _y_ . (These words need not be aligned.)

_• Morpheme_ features: We include indicator
features for all ( _w,mc_ ) pairs, where _w_ is a word
of the source phrase _x_, and _m_ is a possible tense,
person, or number for the target phrase _y_ (drawn
from Table 1). For example, _m_ might be 1st
(first person) or SPre (simple present).

_• Prefix_ and _suffix_ features: For each word or
morpheme feature that fires, 8 backoff features
also fire, where the source word and (if present)
the target word are replaced by their first or last
_i_ characters, for _i_ _∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ .
These templates yield about 4600 features in all, so
the knowledge state has _d_ _≈_ 4600 dimensions.


**5.3** **Learning Models**


We now turn to the question of modeling how the
student’s knowledge state changes during their
session. _**θ**_ _t_ denotes the state at the start of round _t_ .
We take _**θ**_ 1 = **0** and assume that the student uses a
deterministic update rule of the following form: [4]


_**θ**_ _t_ +1 = _**β**_ _t ⊙_ _**θ**_ _t_ + _**α**_ _t ⊙_ **u** _t_ (2)


where **u** _t_ is an _update vector_ that depends on the
student’s experience ( _at,yt,a_ _[′]_ _t_ [)][ at round] _[ t]_ [.]
Why this form? First imagine that the student
is learning by stochastic gradient descent on some
L2-regularized loss function _C· ∥_ _**θ**_ _∥_ [2] + [�] _t_ _[L][t]_ [(] _**[θ]**_ [)][.]
This algorithm’s update rule has the simplified form


_**θ**_ _t_ +1 = _βt_ _·_ _**θ**_ _t_ + _αt_ _·_ **u** _t_ (3)


4Since learning is not perfectly predictable, it would
be more realistic to compute _**θ**_ _t_ by a stochastic update—or
equivalently, by a deterministic update that also depends on a
random noise vector _**ϵ**_ _t_ (which is drawn from, say, a Gaussian).
These noise vectors are “nuisance parameters,” but rather than
integratingovertheirpossiblevalues, astraightforwardapproximation is to optimize them by gradient descent—along with the
other update parameters—so as to locally maximize likelihood.



where **u** _t_ = _−∇Lt_ ( _**θ**_ ) is the steepest-descent
direction on example _t_, _αt >_ 0 is the learning rate at
time _t_, and _βt_ =1 _−αtC_ handles the weight decay
due to following the gradient of the regularizer.
Adaptive versions of stochastic gradient
descent—such as AdaGrad (Duchi et al., 2011) and
AdaDelta (Zeiler, 2012)—are more like our full
rule (2) in that they allow different learning rates
for different parameters.
In general, we can regard _**α**_ _t ∈_ (0 _,_ 1) _[d]_ as modeling
the rates at which the learner updates the various parameters according to **u** _t_, and _**β**_ _t ∈_ (0 _,_ 1) _[d]_ as modeling the rates at which those parameters are forgotten.
These vectors correspond respectively to the _input_
_gates_ and _forget gates_ in recurrent neural network
architectures such as the LSTM (Hochreiter and
Schmidhuber, 1997) or GRU (Cho et al., 2014). As
inthosearchitectures, wewilluseneuralnetworksto
choose _**α**_ _t,_ _**β**_ _t_ at each time step _t_, so that they may be
sensitive in nonlinear ways to the context at round _t_ .


**5.3.1** **Schemes for the Update Vector u** _t_

We assume that **u** _t_ is the gradient of some logprobability, so that the student learns by trying to
increase the log-probability of the correct answer.
However, the student does not always _observe_ the
correct answer _y_ . For example, there is no output
label provided when the student only receives
feedback that their answer is incorrect. Even in such

cases, the student can change their knowledge state.
In this section, we define schemes for defining
**u** _t_ from the experience ( _at,yt,a_ _[′]_ _t_ [)][ at round] _[ t]_ [. Recall]
that _at_ =( _xt,Ot_ ). We omit the _t_ subscripts below.
Suppose the student is told that a particular
phrase _y ∈O_ is the correct translation of _x_ (via an
EX card or via feedback on an answer to an MC or

TP card). Then an apt strategy for the student would
be to use the following gradient: [5]


**∆** = _∇_ _**θ**_ log _p_ ( _y_ _|_ _x,O_ ; _**θ**_ ) (4)

= _**φ**_ ( _x,y_ ) _−_ � _p_ ( _y_ _[′]_ _|_ _x_ ) _**φ**_ ( _x,y_ _[′]_ )

_y_ _[′]_ _∈O_


If the student is told that _y_ is _incorrect_, an apt strategy is to move probability mass collectively to the
other available options, increasing their total probability, since one of those options _must_ be correct.


5An objection is that for an EX or TP card, the student may
not actually know the exact set of options _O_ in the denominator.
We attempted setting _O_ to be the set of English phrases the
student has seen prior to the current question. Though intuitive,
this setting performed worse on all the update and gating
schemes.



241


We call this the _redistribution gradient (RG)_ :


**∆** = _∇_ _**θ**_ log _p_ ( _O−{y}|_ _x,O_ ; _**θ**_ ) (5)


= =
� _p_ ( _y_ _[′]_ _|_ _x,y_ _[′]_ _̸_ _y_ ) _**φ**_ ( _x,y_ _[′]_ ) (6)

_y_ _[′]_ _∈O−{y}_

_−_ � _p_ ( _y_ _[′]_ _|_ _x_ ) _**φ**_ ( _x,y_ _[′]_ )

_y_ _[′]_ _∈O_


where _p_ ( _y_ _[′]_ _|_ _x,y_ _[′]_ = _̸_ _y_ ) is a renormalized distribution
over just the options _y_ _[′]_ _∈O−{y}_ . Note that if the
student selects two wrong answers _y_ 1 _,y_ 2 in a row
on an MC card, the first update will subtract the
average features of _O_ and add those of _O −{y_ 1 _}_ ;
the second update will subtract the average features
of _O −{y_ 1 _}_ and add those of _O −{y_ 1 _,y_ 2 _}_ . The
intermediate addition and subtraction cancel out if

the same _**α**_ vector is used at both rounds, so the net
effect is to shift probability mass from the 5 initial
options to the 3 remaining ones. [6]

An alternate scheme for incorrect _y_ is to use
_−_ **∆** . We call this _negative gradient (NG)_ .
Since the RG and NG update vectors both worked
well for handling incorrect _y_, we also tried linearly
interpolating them _(RNG)_, with **u** _t_ = _**γ**_ _t ⊙_ **∆** +
( **1** _−_ _**γ**_ _t_ ) _⊙−_ **∆** . The interpolation vector _**γ**_ _t_ has elementsin (0 _,_ 1), andmaydependonthecontext(possibly different for MC and EX cards, for example).
Finally, the _feature vector (FG)_ scheme simply
adds the features _φ_ ( _x, y_ ) when _y_ is correct or
subtracts them when _y_ is incorrect. This is
appropriate for a student who pays attention only
to _y_, without bothering to note that the alternative
options in _O_ are (respectively) incorrect or correct.
Recall from section 3.1 that the system sometimes gives both indicative and explicit feedback,
telling the student that one phrase is incorrect and
a different phrase is correct. We treat these as two
successive updates with update vectors **u** _t_ and **u** _t_ +1.
Notice that in the FG scheme, adding this pair of
update vectors resembles a perceptron update.
Table 2 summarizes our update schemes.


**5.3.2** **Schemes for the Gates** _**α**_ _t,_ _**β**_ _t,_ _**γ**_ _t_

We characterize each update _t_ by a 7-dimensional
context vector **c** _t_, which summarizes what the
student has experienced. The first three elements
in **c** _t_ are binary indicators of the type of flash card


6Arguably, a zeroth update should be allowed as well: upon
first viewing the MC card, the student should have the chance
to subtract the average features of the full set of possibilities
and add those of the 5 options in _O_, since again, the system
is implying that one of those 5 options _must_ be correct.



_̸_


_̸_


where we take _p_ ( _yt | ···_ ) = 1 at steps where the
student makes no response (EX cards and explicit



Update Scheme Correct Incorrect


redistribution (RG) **u** _t_ = **∆** **u** _t_ = **∆**

negative grad. (NG) **u** _t_ = **∆** **u** _t_ = _−_ **∆**

feature vector (FG) **u** _t_ = _**φ**_ ( _x,y_ ) **u** _t_ = _−_ _**φ**_ ( _x,y_ )

_̸_


Table 2: Summary of update schemes (other than RNG).


(EX, MC or TP). The next three elements are binary
indicators of the type of information that caused the
update: correct student answer, incorrect student

_̸_

answer, or revealed answer (via an EX card or
explicit feedback). As a reminder, the system can
respond with an indication that the answer is correct
or incorrect, or it can reveal the answer. Finally, the
last element of **c** _t_ is 1 _/|O|_, the chance probability
of success on this card. From **c** _t_, we define


_**α**_ _t_ = _σ_ ( **W** _[α]_ **c** _t_ + _b_ _[α]_ **1** ) _∈_ (0 _,_ 1) _[d]_ (7)

_**β**_ _t_ = _σ_ ( **W** _[β]_ **c** _t−_ 1 + _b_ _[β]_ **1** ) _∈_ (0 _,_ 1) _[d]_ (8)

_**γ**_ _t_ = _σ_ ( **W** _[γ]_ **c** _t_ + _b_ _[γ]_ **1** ) _∈_ (0 _,_ 1) _[d]_ (9)


where **c** 0 = **0** . Each gate vector is now parameterized by a weight matrix **W** _∈_ R _[d][×]_ [7], where _d_ is the
dimensionality of the gradient and knowledge state.
We also tried simpler versions of this model. In
the _vector model (VM)_, we define _**α**_ _t_ = _σ_ ( **b** _[α]_ ), and
_**β**_ _t,_ _**γ**_ _t_ similarly. These vectors do not vary with time
and simply reflect that some parameters are more
labile than others. Finally, the _scalar model (SM)_ defines _**α**_ _t_ = _σ_ ( _b_ _[α]_ **1** ), so that all parameters are equally
labile. One could also imagine tying the gates for
features derived from the same template, meaning
that some _kinds_ of features (in some contexts) are
more labile than others, or reducing the number of
parameters by learning low-rank **W** matrices.
While we also tried augmenting the context
vector **c** _t_ with the knowledge state _**θ**_ _t_, this resulted
in far too many parameters to train well, and did not
help performance in pilot tests.


**5.4** **Parameter Estimation**


We tune the **W** and _b_ parameters of the model by
maximum likelihood, so as to better predict the
students’ responses _yt_ . The likelihood function is



_̸_


_̸_


_p_ ( _y_ 1 _,...yT |_ _at,...aT_ )=


=



_̸_


_̸_


_T_
� _p_ ( _yt |_ _a_ 1: _t,y_ 1: _t−_ 1 _,a_ _[′]_ 1: _t−_ 1 [)]


_t_ =1


_T_
� _p_ ( _yt |_ _at_ ; _**θ**_ _t_ ) (10)


_t_ =1



_̸_


_̸_


242


feedback). Note that the model assumes that _**θ**_ _t_ is a
sufficient statistic of the student’s past experiences.
For each (update scheme, gating scheme)
combination, we trained the parameters using SGD
with RMSProp updates (Tieleman and Hinton,
2012) to maximize the regularized log-likelihood

� log _p_ ( _yt |_ _xt_ ; _**θ**_ _t_ ) _−C·∥_ **W** _∥_ [2] (11)

_t,τt_ =0


summed over all students. Note that _**θ**_ _t_ depends on
the parameters through the gated update rule (2).
The development set was used for early stopping
and to tune the regularization parameter _C_ . [7]


**6** **Data Collection**


We recruited 153 unique “students” via Amazon Mechanical Turk (MTurk). MTurk participants were
compensated $1 for completing the training and test
sessions and a bonus of $10 was given to the three
top scoring students. In our dataset, we retained
only the 121 students who answered all questions.


**6.1** **Language Obfuscation**


Fig. 1 shows a few example flash cards for a native
English speaker learning Spanish. Fig. 1 shows
all our Spanish-English phrase pairs. In our actual
task, however, we invented an artificial language
for the MTurk students to learn, which allowed us
to ignore the problem of students with different
initial knowledge levels. We generated our artificial
language by enciphering the Spanish orthographic
representations. We created a mapping from the true
source string alphabet to an alternative, manually
defined alphabet, while attempting to preserve pronounceability (by mapping vowels to vowels, etc.).
For example, mirar was transformed into melil
and t´u aceptas became pi icedpiz.


**6.2** **Card Ordering Policy**


In the future, we expect to use planning or reinforcement learning to choose the sequence of stimuli
for the student. For the present study of student
behavior, however, we hand-designed a simple
stochastic policy for choosing the stimuli.
The policy must decide what foreign phrase and
card modality to use at each training step. Our policy likes to repeat phrases with which participants


7
We searched _C ∈{_ 0 _._ 00025, 0 _._ 0005, 0 _._ 001 _, ..._, 0 _._ 01,
0 _._ 025, 0 _._ 05, 0 _._ 1 _}_ for each gating model and update scheme
combination. _C_ =0 _._ 0025 gave best results for the CM models,
0 _._ 01 for VM and 0 _._ 0005 for SM.



On each round, we sample a phrase _x_ from either _Pv_ or _Pe_ (equal probability); these distributions are computed by applying a softmax _g_ ( _._ )
over the vectors **v** and **e** respectively (see Eq. 12).
Once the phrase _x_ is decided, the modality (EX,
MC, TP) is chosen stochastically using probabilities (0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 4), except that probabilities (1 _,_ 0 _,_ 0)
are used for the first example of the session, and
(0 _._ 4 _,_ 0 _._ 6 _,_ 0) if _x_ is not “TP-qualified.” A phrase is
TP-qualified if the student has seen both _x_ ’s pronoun
and _x_ ’s verb lemma on previous cards (even if their
correct translation was not revealed). For an MC
card, the distractor phrases are sampled uniformly
without replacement from the 38 other phrases.


**7** **Results & Experiments**


We partitioned the students into three groups: 80
students for training, 20 for development, and 21
for testing. Most students found the task difficult;
the average score on the 7-question quiz—was
2 _._ 81 correct, with maximum score of 6. (Recall
from section 3.2 that the quiz questions were typing
questions, not multiple choice questions.)
After constructing each model, we evaluated it
on the held-out data: the 728 responses from the
21 testing students. We measure the log-probability
under the model of each actual response (“crossentropy”), and also the fraction of responses that


8
Arguably we should have updated _ex_ instead by
adding/subtracting 1, since it will be exponentiated later.



had trouble—in hopes that these already-taught
phrases are on the verge of being learned. It also
likes to pick out new phrases. This was inspired by
the popular Leitner (1972) approach, which devised
a system of buckets that control how frequently an
item is reviewed by a student. Leitner proposed
buckets with review frequency rates of every day,
every 2 days, every 4 days and so on.
For each foreign phrase _x ∈X_, we maintain a
_novelty score vx_, which is a function of the number
of times the phrase is exposed to a student and an
_error score ex_, which is a function of the number
of times the student incorrectly responded to the
phrase. These scores are initialized to 1 and updated
as follows: [8]


_vx ←_ _vx_ _−_ 1 when _x_ is viewed



_ex ←_



2 _ex_ when student gets _x_ wrong
0 _._ 5 _ex_ when student gets _x_ right
�



_x_ _∼_ _[g]_ [(] **[v]** [)][+] _[g]_ [(] **[e]** [)] (12)

2



243


were correctly predicted if our prediction was the
model’s max-probability response (“accuracy”).
Table 3 shows the results of our experiment. All
of our models were predictive, doing far better than
a uniform baseline that assigned equal probability
1 _/|O|_ to all options. Our best models are shown in
the final two lines, RNG+VM and RNG+CM.
Which update scheme was best? Interestingly,
although the RG update vector is principled from a
_machine_ learning viewpoint, the NG update vector
sometimes achieved better accuracy—though
worse perplexity—when predicting the responses
of _human_ learners. [9] We got our best results on both
metrics by interpolating between RG and NG (the
RNG scheme). Recall that the NG scheme was
motivated by the notion that students who guessed
wrong may not study the alternative answers (even
though one is correct), either because it is too much
trouble to study them or because (for a TP card)
those alternatives are not actually shown.
Which gating mechanism was best? In almost all
cases, we found that more parameters helped, with
CM _>_ VM _>_ SM on accuracy, and a similar pattern
on cross-entropy (with VM sometimes winning
but only slightly). In short, it helps to use different
learning rates for different features, and it probably
helps to make them sensitive to the learning context.
Surprisingly, the simple FG scheme outperformed both RG and NG when used in conjunction
with a scalar retention and acquisition gate. This,
however, did not extend to more complex gates.
Fig. 2 shows a breakdown of the prediction
accuracy measures according to whether the card
was MC or TP, and according to whether the
student’s answer was correct (C) or incorrect (IC).
Unsurprisingly, all the models have an easier time
predicting the student’s guess when the student is
correct, since the predicted parameters _**θ**_ _t_ will often
pick the correct answer. However, this is where the
vector and context gates far outperform the scalar
gates. All the models find predicting the incorrect
answers of the students difficult. Moreover, when
predicting these incorrect answers, the RG models
do slightly better than the NG models.
The models obviously have higher accuracy
when predicting student answers for MC cards
than for TP cards, as MC cards have fewer options.
Again, within both of these modalities, the vector
and context gates outperform the scalar gate.


9Even the FG vector sometimes won (on both metrics!),
but this happened only with the worst gating mechanism, SM.



For MC cards, the chance probability is in _{_ 5 _[,]_ 4 _[,]_ 3 _[}]_ [—]

depending on how many options remain—while for TP cards
it is 1
39 [.]



Update Scheme Gating Mechanism accuracy cross-ent.


(Uniform baseline) 0.133 2.459


FG SM 0.239 _[∗]_ 2.362
FG VM 0.357 _[†]_ 2.130

FG CM 0.401 2.025


RG SM 0.135 3.194
RG VM 0.397 _[†]_ 1.909

RG CM 0.405 1.938


NG SM 0.185 _[∗]_ 4.674
NG VM 0.394 _[†]_ 2.320
NG CM 0.449 _[†∗]_ 2.244


RNG (mixed) SM 0.183 3.502
RNG (mixed) VM 0.427 1.855
RNG (mixed) CM 0.449 1.888


Table 3: Table summarizing prediction accuracy and crossentropy (in nats per prediction) for different models. Larger
accuracies and smaller cross-entropies are better. Within
an update scheme, the _[†]_ indicates significant improvement
(McNemar’s test, _p <_ 0 _._ 05) over the next-best gating
mechanism. Within g a gating mechanism, the _[∗]_ indicates
significant improvement over the next-best update scheme. For
example, NG+CM is significantly better than NG+VM, so it
receives a _[†]_ ; it is also significantly better than RG+CM, and
receives a _[∗]_ as well. These comparisons are conducted only
among the pure update schemes (above the double line). All
other models are significantly better than RG+SM ( _p<_ 0 _._ 01).


Finally, Fig. 3 examines how these models behave
when making specific predictions over a training
sequence for a single student. At each step we plot
the difference in log-probability between our model
and a uniform baseline model. Thus, a marker above
0 means that our model assigned the student’s answer a probability higher than chance. [10] To contrast the performance difference, we show both the
highest-accuracy model (RNG+CM) and the lowestaccuracy model (RG+SM). For a high-scoring student (Fig. 3a), we see RNG+CM has a large margin
over RG+SM and a slight upward trend. A higher
probability than chance is noticeable even when the
student makes mistakes (indicated by hollow markers). In contrast, for an average student (Fig. 3b), the
margin between the two models is less perceptible.
While the CM+NG model is still above the SM+RG

line, there are some answers where CM+NG does
very poorly. This is especially true for some of the
wrong answers, for example at training steps 25, 29
and 33. Upon closer inspection into the model’s error in step 33, we found the prompt received at this
training step was ekki mel¨u as a MC card, which
had been shown to the student on three prior occasions, and the student even answered correctly on
one of these occasions. This explains why the model



10For MC cards, the chance probability is in _{_ 1




[1] [1]

4 _[,]_ 3



1 [1]

5 _[,]_ 4



244


0 10 15 20 25 30 35 40

training steps



0 10 15 20 25 30 35 40 45

training steps



(a) a student with quiz score 6/7 (b) a student with quiz score 2/7

Figure 3: Predicting a specific student’s responses. For each response, the plot shows our model’s improvement in log-probability
over the uniform baseline model. TP cards are the square markers connected by solid lines (the final 7 squares are the quiz), while
MC cards—which have a much higher baseline—are the circle markers connected by dashed lines. Hollow and solid markers
indicate correct and incorrect answers respectively. The RNG+CM model is shown in blue and the FG+SM model in red.



was surprised to see the student make this error.


**7.1** **Comparison with Less Restrictive Model**


Our parametric knowledge tracing architecture
models the student as a typical structured prediction
system, which maintains weights for hand-designed
features and updates them roughly as an online
learning algorithm would. A natural question
is whether this restricted architecture sacrifices
performance for interpretability, or improves
performance via useful inductive bias.
To consider the other end of the spectrum, we
implemented a flexible LSTM model in the style
of recent deep learning research. This alternative
model predicts each response by a student (i.e., on
an MC or TP card) given the entire history of previous interactions with that student as summarized

by an LSTM. The LSTM architecture is formally
capable of capturing update rules exactly like those
of PKT, but it is far from limited to such rules.
Much like equation (1), at each time _t_ we predict



exp( **h** _t_ _·_ _**ψ**_ ( _y_ ))
_p_ ( _yt_ = _y_ _|_ _at_ )= (13)
~~�~~ _y_ _[′]_ _∈Ot_ [exp(] **[h]** _[t]_ _[·][ψ]_ [(] _[y]_ [))]


for each possible response _y_ in the set of options



_Ot_, where _**ψ**_ ( _y_ ) _∈_ R _[d]_ is a learned embedding of
response _y_ . Here **h** _t ∈_ R _[d]_ denotes the hidden state
of the LSTM, which evolves as the student interacts
withthesystemandlearns. **h** _t_ dependsontheLSTM
inputs for all times _< t_, just like the knowledge
state _**θ**_ _t_ in equations (1)–(2). It also depends on the
LSTM input for time _t_, since that specifies the flash
card _at_ to which we are predicting the response _yt_ .
Each flash card _a_ = ( _x, O_ ) is encoded by a
concatenation **a** of three vectors: a one-hot 39
dimensional vector specifying the foreign phrase
_x_, a 39-dimensional binary vector _**O**_ indicating the
possible English options in _O_, and a one-hot vector
indicating whether the card is EX, MC, or TP.


When reading the history of past interactions, the
LSTM input at each time step _t_ concatenates the vector representation **a** _t_ of the current flash card with
vectors **a** _t−_ 1 _,_ **y** _t−_ 1 _,_ **f** _t−_ 1 that describe the student’s
experience in round _t −_ 1: these respectively encode the previous flash card, the student’s response
to it (a one-hot 39-dimensional vector), and the resulting feedback (a 39-dimensional binary vector
that indicates the remaining options after feedback).
Thus, if the student receives no feedback, then
**f** _t−_ 1 = _**O**_ _t−_ 1. Indicative feedback sets **f** _t−_ 1 = **y** _t−_ 1



245


Model Parameters Accuracy(test) Cross-Entropy


RNG+CM _≈_ 97K 0.449 1.888

LSTM _≈_ 25K 0.429 1.992


Table 4: Comparison of our best-performing PKT model
(RNG+CM) to our LSTM model. On our dataset, PKT outperforms the LSTM both in terms of accuracy and cross-entropy.


or **f** _t−_ 1 = _**O**_ _t−_ 1 _−_ **y** _t_, according to whether the student was correct or incorrect. Explicit feedback
(including for an EX card) sets **f** _t−_ 1 to a one-hot representation of the correct answer. Thus, **f** _t−_ 1 gives
the set of “positive” options that we used in the RG
update vector, while _**O**_ _t−_ 1 gives the set of “negative”
options, allowing the LSTM to similarly update its
hidden state from **h** _t−_ 1 to **h** _t_ to reflect learning. [11]

As in section 5.4, we train the parameters by
L2-regularized maximum likelihood, with early
stopping on development data. The weights for
the LSTM were initialized uniformly at random
_∼_ _U_ ( _−δ,_ + _δ_ ), where _δ_ = 0 _._ 01, and RMSProp
was used for gradient descent. We settled on a
regularization coefficient of 0 _._ 002 after a line search.
The number of hidden units _d_ was also tuned using
line search. Interestingly, a dimensionality of just
_d_ =10 performed best on dev data: [12] at this size, the
LSTM has _fewer_ parameters than our best model.

The result is shown in Table 4. These results favor

our restricted PKT architecture. We acknowledge
that the LSTM might perform better when a larger
training set was available (which would allow a
larger hidden layer), or using a different form of
regularization (Srivastava et al., 2014).
Intermediate or hybrid models would of course
also be possible. For example, we could predict
_p_ ( _y | at_ ) via (1), defining _**θ**_ _t_ as **h** _[⊤]_ _t_ _[M]_ [, a learned]
linear function of _ht_ . This variant would again have
access to our hand-designed features _**φ**_ ( _x,y_ ), so
that it would know which flash cards were similar.
In fact _**θ**_ _t_ _·_ _**φ**_ ( _x,y_ ) in (1) equals **h** _t_ _·_ ( _M_ _**φ**_ ( _x,y_ )), so


11This architecture is formally able to mimic PKT. We would
store _**θ**_ in the LSTM’s vector of cell activations, and configure
the LSTM’s “input” and “forget” gates to update this according
to (2) where **u** _t_ is computed from the input. Observe that each
feature in section 5.2 has the form _φij_ ( _x,y_ ) = _ξi_ ( _x_ ) _·_ _ψj_ ( _y_ ).
Consider the hidden unit in **h** corresponding to this feature,
with activation _θij_ . By configuring this unit’s “output” gate
to be _ξi_ ( _x_ ) (where _x_ is the current foreign phrase given in the
input), we would arrange for this hidden unit to have output
_ξi_ ( _x_ ) _·θij_, which will be multiplied by _ψj_ ( _y_ ) in (13) to recover
_θij ·φij_ ( _x,y_ ) just as in (1). (More precisely, the output would
be sigmoid( _ξi_ ( _x_ ) _·_ _θij_ ), but we can evade this nonlinearity if
we take the cell activations to be a scaled-down version of _**θ**_
and scale up the embeddings _**ψ**_ ( _y_ ) to compensate.)
12
We searched 0 _._ 001 _,_ 0 _._ 002 _,_ 0 _._ 005 _,_ 0 _._ 01 _,_ 0 _._ 02 _,_ 0 _._ 05 for the
regularization coefficient, and 5 _,_ 10 _,_ 15 _,_ 20 _,_ 50 _,_ 100 _,_ 200 for the
number of hidden units.



_M_ can be regarded as projecting _**φ**_ ( _x,y_ ) down to
the LSTM’s hidden dimension _d_, learning how to
weight and use these features. In this variant, the
LSTM would no longer need to take _at_ as part of
its input at time _t_ : rather, **h** _t_ (just like _**θ**_ _t_ in PKT)
would be a pure representation of the student’s
knowledge state at time _t_, capable of predicting
_yt_ for _any at_ . This setup more closely resembles
PKT—or the DKT LSTM of Piech et al. (2015).
Unlike the DKT paper, however, it would still
predict the student’s specific response, not merely
whether they were right or wrong.


**8** **Conclusion**


We have presented a cognitively plausible model
that traces a human student’s knowledge as he or she
interacts with a simple online tutoring system. The
student must learn to translate very short inflected
phrases from an unfamiliar language into English.
Our model assumes that when a student recalls or

guesses the translation, he or she is attempting to
solve a structured prediction problem of choosing
the best translation, based on salient features of the
input-output pair. Specifically, we characterize the
student’s knowledge as a vector of feature weights,
which is updated as the student interacts with the system. While the phrasal features memorize the translations of entire input phrases, the other features can
pick up on the translations of individual words and
sub-words, which are reusable across phrases.
We collected and modeled human-subjects
data. We experimented with models using several
different update mechanisms, focusing on the
student’s treatment of negative feedback and the
degree to which the student tends to update or
forget specific weights in particular contexts. We
also found that in comparison to a less constrained
LSTM model, we can better fit the human behavior
by using weight update schemes that are broadly
consistent with schemes used in machine learning.
In the future, we plan to experiment with more
variants of the model, including variants that allow
noiseandpersonalization. Mostimportant, wemean
tousethemodelforplanningwhichflashcards, feedback, or other stimuli to show next to a given student.


**Acknowledgments**


This material is based upon work supported by a
seed grant from the Science of Learning Institute
at Johns Hopkins University.



246


**References**


Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
Schwenk, and Yoshua Bengio. 2014. Learning
phrase representations using rnn encoder–decoder
for statistical machine translation. In _Proceedings of_
_the 2014 Conference on Empirical Methods in Natu-_
_ral Language Processing (EMNLP)_ . Association for
Computational Linguistics, Doha, Qatar, pages 1724–
1734. http://www.aclweb.org/anthology/D14-1179.


Albert T Corbett and John R Anderson. 1994. Knowledge tracing: Modeling the acquisition of procedural
knowledge. _User modeling and user-adapted_
_interaction_ 4(4):253–278.


John Duchi, Elad Hazan, and Yoram Singer. 2011.
Adaptive subgradient methods for online learning
and stochastic optimization. _Journal of Machine_
_Learning Research_ 12(Jul):2121–2159.


Jos´e Gonz´alez-Brenes, Yun Huang, and Peter
Brusilovsky. 2014. General features in knowledge
tracing to model multiple subskills, temporal item response theory, and expert knowledge. In _Proceedings_
_of the 7th International Conference on Educational_
_Data Mining_ . University of Pittsburgh, pages 84–91.


Sepp Hochreiter and J¨urgen Schmidhuber. 1997.
Long short-term memory. _Neural computation_
9(8):1735–1780.


J. J. Hopfield. 1982. Neural networks and physical
systems with emergent collective computational
abilities. In _Proceedings of the National Academy of_
_Sciences of the USA_ . volume 79, pages 2554–2558.


Yun Huang, J Guerra, and Peter Brusilovsky. 2016.
Modeling skill combination patterns for deeper
knowledge tracing. In _Proceedings of the 6th Work-_
_shop on Personalization Approaches in Learning_
_Environments (PALE 2016)_ . 24th Conference on User
Modeling, Adaptation and Personalization, Halifax,
Canada.


Mohammad Khajah, Rowan Wing, Robert Lindsey,
and Michael Mozer. 2014a. Integrating latent-factor
and knowledge-tracing models to predict individual
differences in learning. In _Proceedings of the 7th In-_
_ternational Conference on Educational Data Mining_ .


Mohammad M Khajah, Yun Huang, Jos´e P Gonz´alezBrenes, Michael C Mozer, and Peter Brusilovsky.
2014b. Integrating knowledge tracing and item
response theory: A tale of two frameworks. In
_Proceedings of Workshop on Personalization Ap-_
_proaches in Learning Environments (PALE 2014) at_
_the 22th International Conference on User Modeling,_
_Adaptation,_ _and Personalization_ . University of
Pittsburgh, pages 7–12.


K. R. Koedinger, P. I. Pavlick Jr., J. Stamper, T. Nixon,
and S. Ritter. 2011. Avoiding problem selection
thrashing with conjunctive knowledge tracing. In
_Proceedings of the 4th International Conference on_


247



_Educational Data Mining_ . Eindhoven, NL, pages
91–100.


Jung In Lee and Emma Brunskill. 2012. The impact
on individualizing student models on necessary
practice opportunities. _International Educational_
_Data Mining Society_ .


Sebastian Leitner. 1972. _So lernt man lernen: der Weg_
_zum Erfolg_ . Herder, Freiburg.


Chris Piech, Jonathan Bassen, Jonathan Huang, Surya
Ganguli, Mehran Sahami, Leonidas J Guibas, and
Jascha Sohl-Dickstein. 2015. Deep knowledge tracing. In _Advances in Neural Information Processing_
_Systems_ . pages 505–513.


Michael I Posner. 1989. _Foundations of cognitive_
_science_ . MIT press Cambridge, MA.


D. L. Schacter. 1989. Memory. In M. I. Postner, editor,
_Foundations of Cognitive Science_, MIT Press, pages
683–725.


Burr Settles and Brendan Meeder. 2016. A trainable
spaced repetition model for language learning. In
_Proceedings of the 54th Annual Meeting of the_
_Association for Computational Linguistics (Volume_
_1:_ _Long Papers)_ . Association for Computational
Linguistics, Berlin, Germany, pages 1848–1858.
http://www.aclweb.org/anthology/P16-1174.


Paul Smolensky. 1986. Information processing in
dynamical systems: Foundations of harmony theory.
In D. E. Rumelhart, J. L. McClelland, and the
PDP Research Group, editors, _Parallel Distributed_
_Processing: Explorations in the Microstructure of_
_Cognition_, MIT Press/Bradford Books, Cambridge,
MA, volume 1: Foundations, pages 194–281.


Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
Ilya Sutskever, and Ruslan Salakhutdinov.
2014. Dropout: A simple way to prevent
neural networks from overfitting. _Journal_
_of_ _Machine_ _Learning_ _Research_ 15:1929–1958.
http://jmlr.org/papers/v15/srivastava14a.html.


Tijmen Tieleman and Geoffrey Hinton. 2012. Lecture
6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. _COURSERA: Neural_
_networks for machine learning_ 4(2).


Yanbo Xu and Jack Mostow. 2012. Comparison of
methods to trace multiple subskills: Is LR-DBN best?
In _Proceedings of the 5th International Conference_
_on Educational Data Mining_ . pages 41–48.


Matthew D Zeiler. 2012. Adadelta: an adaptive learning
rate method. _arXiv preprint arXiv:1212.5701_ .


