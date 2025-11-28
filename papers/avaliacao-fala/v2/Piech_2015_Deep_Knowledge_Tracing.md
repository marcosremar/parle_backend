# **Deep Knowledge Tracing**

**Chris Piech** _[∗]_ **, Jonathan Bassen** _[∗]_ **, Jonathan Huang** _[∗‡]_ **, Surya Ganguli** _[∗]_ **,**
**Mehran Sahami** _[∗]_ **, Leonidas Guibas** _[∗]_ **, Jascha Sohl-Dickstein** _[∗†]_

_∗_ Stanford University, _†_ Khan Academy, _‡_ Google
_{_ piech,jbassen _}_ @cs.stanford.edu, jascha@stanford.edu,


**Abstract**


Knowledge tracing—where a machine models the knowledge of a student as they
interact with coursework—is a well established problem in computer supported
education. Though effectively modeling student knowledge would have high educational impact, the task has many inherent challenges. In this paper we explore
the utility of using Recurrent Neural Networks (RNNs) to model student learning.
The RNN family of models have important advantages over previous methods
in that they do not require the explicit encoding of human domain knowledge,
and can capture more complex representations of student knowledge. Using neural networks results in substantial improvements in prediction performance on a
range of knowledge tracing datasets. Moreover the learned model can be used for
intelligent curriculum design and allows straightforward interpretation and discovery of structure in student tasks. These results suggest a promising new line of
research for knowledge tracing and an exemplary application task for RNNs.


**1** **Introduction**


Computer-assisted education promises open access to world class instruction and a reduction in the
growing cost of learning. We can develop on this promise by building models of large scale student
trace data on popular educational platforms such as Khan Academy, Coursera, and EdX.


Knowledge tracing is the task of modelling student knowledge over time so that we can accurately
predict how students will perform on future interactions. Improvement on this task means that resources can be suggested to students based on their individual needs, and content which is predicted
to be too easy or too hard can be skipped or delayed. Already, hand-tuned intelligent tutoring systems that attempt to tailor content show promising results [28]. One-on-one human tutoring can
produce learning gains for the average student on the order of two standard deviations [5] and machine learning solutions could provide these benefits of high quality personalized teaching to anyone
in the world for free. The knowledge tracing problem is inherently difficult as human learning is
grounded in the complexity of both the human brain and human knowledge. Thus, the use of rich
models seems appropriate. However most previous work in education relies on first order Markov
models with restricted functional forms.


In this paper we present a formulation that we call Deep Knowledge Tracing (DKT) in which we
apply flexible recurrent neural networks that are ‘deep’ in time to the task of knowledge tracing. This
family of models represents latent knowledge state, along with its temporal dynamics, using large
vectors of artificial ‘neurons’, and allows the latent variable representation of student knowledge to
be learned from data rather than hard-coded. The main contributions of this work are:


1. A novel way to encode student interactions as input to a recurrent neural network.

2. A 25% gain in AUC over the best previous result on a knowledge tracing benchmark.

3. Demonstration that our knowledge tracing model does not need expert annotations.

4. Discovery of exercise influence and generation of improved exercise curricula.


1


E[ _p_ ]





Exercise attempted:   correct,   incorrect


10 20 30 40 50





Exercise index


_Figure 1: A single student and her predicted responses as she solves 50 Khan Academy exercises. She seems to_
_master finding x and y intercepts and then has trouble transferring knowledge to graphing linear equations._


The task of knowledge tracing can be formalized as: given observations of interactions **x** 0 _. . ._ **x** _t_
taken by a student on a particular learning task, predict aspects of their next interaction **x** _t_ +1 [6].
In the most ubiquitous instantiation of knowledge tracing, interactions take the form of a tuple of
**x** _t_ = _{qt, at}_ that combines a tag for the exercise being answered _qt_ with whether or not the exercise
was answered correctly _at_ . When making a prediction, the model is provided the tag of the exercise
being answered, _qt_ and must predict whether the student will get the exercise correct, _at_ . Figure 1
shows a visualization of tracing knowledge for a single student learning 8th grade math. The student
first answers two square root problems correctly and then gets a single x-intercept exercise incorrect.
In the subsequent 47 interactions the student solves a series of x-intercept, y-intercept and graphing
exercises. Each time the student answers an exercise we can make a prediction as to whether or not
she would answer an exercise of each type correctly on her next interaction. In the visualization
we only show predictions over time for a relevant subset of exercise types. In most previous work,
exercise tags denote the single “concept” that human experts assign to an exercise. Our model
can leverage, but does not require, such expert annotation. We demonstrate that in the absence of
annotations the model can autonomously learn content substructure.


**2** **Related Work**


The task of modelling and predicting how human beings learn is informed by fields as diverse
as education, psychology, neuroscience and cognitive science. From a social science perspective
learning has been understood to be influenced by complex macro level interactions including affect

[21], motivation [10] and even identity [4]. The challenges present are further exposed on the micro
level. Learning is fundamentally a reflection of human cognition which is a highly complex process.
Two themes in the field of cognitive science that are particularly relevant are theories that the human
mind, and its learning process, are recursive [12] and driven by analogy [13].


The problem of knowledge tracing was first posed, and has been heavily studied within the intelligent
tutoring community. In the face of aforementioned challenges it has been a primary goal to build
models which may not capture all cognitive processes, but are nevertheless useful.


**2.1** **Bayesian Knowledge Tracing**


Bayesian Knowledge Tracing (BKT) is the most popular approach for building temporal models
of student learning. BKT models a learner’s latent knowledge state as a set of binary variables,
each of which represents understanding or non-understanding of a single concept [6]. A Hidden
Markov Model (HMM) is used to update the probabilities across each of these binary variables, as a
learner answers exercises of a given concept correctly or incorrectly. The original model formulation
assumed that once a skill is learned it is never forgotten. Recent extensions to this model include
contextualization of guessing and slipping estimates [7], estimating prior knowledge for individual
learners [33], and estimating problem difficulty [23].


With or without such extensions, Knowledge Tracing suffers from several difficulties. First, the
binary representation of student understanding may be unrealistic. Second, the meaning of the
hidden variables and their mappings onto exercises can be ambiguous, rarely meeting the model’s
expectation of a single concept per exercise. Several techniques have been developed to create and
refine concept categories and concept-exercise mappings. The current gold standard, Cognitive Task
Analysis [31] is an arduous and iterative process where domain experts ask learners to talk through


2


their thought processes while solving problems. Finally, the binary response data used to model
transitions imposes a limit on the kinds of exercises that can be modeled.


**2.2** **Other Dynamic Probabilistic Models**


Partially Observable Markov Decision Processes (POMDPs) have been used to model learner behavior over time, in cases where the learner follows an open-ended path to arrive at a solution [29].
Although POMDPs present an extremely flexible framework, they require exploration of an exponentially large state space. Current implementations are also restricted to a discrete state space,
with hard-coded meanings for latent variables. This makes them intractable or inflexible in practice,
though they have the potential to overcome both of those limitations.


Simpler models from the Performance Factors Analysis (PFA) framework [24] and Learning Factors
Analysis (LFA) framework [3] have shown predictive power comparable to BKT [14]. To obtain
better predictive results than with any one model alone, various ensemble methods have been used
to combine BKT and PFA [8]. Model combinations supported by AdaBoost, Random Forest, linear
regression, logistic regression and a feed-forward neural network were all shown to deliver superior
results to BKT and PFA on their own. But because of the learner models they rely on, these ensemble
techniques grapple with the same limitations, including a requirement for accurate concept labeling.


Recent work has explored combining Item Response Theory (IRT) models with switched nonlinear
Kalman filters [20], as well as with Knowledge Tracing [19, 18]. Though these approaches are
promising, at present they are both more restricted in functional form and more expensive (due to
inference of latent variables) than the method we present here.


**2.3** **Recurrent Neural Networks**


Recurrent neural networks are a family of flexible dynamic models which connect artificial neurons
over time. The propagation of information is recursive in that hidden neurons evolve based on both
the input to the system and on their previous activation [32]. In contrast to hidden Markov models
as they appear in education, which are also dynamic, RNNs have a high dimensional, continuous,
representation of latent state. A notable advantage of the richer representation of RNNs is their ability to use information from an input in a prediction at a much later point in time. This is especially
true for Long Short Term Memory (LSTM) networks—a popular type of RNN [16].


Recurrent neural networks are competitive or state-of-the-art for several time series tasks–for instance, speech to text [15], translation [22], and image captioning [17]–where large amounts of
training data are available. These results suggest that we could be much more successful at tracing
student knowledge if we formulated the task as a new application of temporal neural networks.


**3** **Deep Knowledge Tracing**


We believe that human learning is governed by many diverse properties – of the material, the context,
the timecourse of presentation, and the individual involved – many of which are difficult to quantify
relying only on first principles to assign attributes to exercises or structure a graphical model. Here
we will apply two different types of RNNs – a vanilla RNN model with sigmoid units and a Long
Short Term Memory (LSTM) model – to the problem of predicting student responses to exercises
based upon their past activity.


**3.1** **Model**


Traditional Recurrent Neural Networks (RNNs) map an input sequence of vectors **x** 1 _, . . .,_ **x** _T_, to
an output sequence of vectors **y** 1 _, . . .,_ **y** _T_ . This is achieved by computing a sequence of ‘hidden’
states **h** 1 _, . . .,_ **h** _T_ which can be viewed as successive encodings of relevant information from past
observations that will be useful for future predictions. See Figure 2 for a cartoon illustration. The
variables are related using a simple network defined by the equations:


**h** _t_ = tanh ( **W** _hx_ **x** _t_ + **W** _hh_ **h** _t−_ 1 + **b** _h_ ) _,_ (1)
**y** _t_ = _σ_ ( **W** _yh_ **h** _t_ + **b** _y_ ) _,_ (2)


3


_Figure 2: The connection between variables in a simple recurrent neural network. The inputs (_ **x** _t) to the_
_dynamic network are either one-hot encodings or compressed representations of a student action, and the_
_prediction (_ **y** _t) is a vector representing the probability of getting each of the dataset exercises correct._


where both tanh and the sigmoid function, _σ_ ( _·_ ), are applied elementwise. The model is parameterized by an input weight matrix **W** _hx_, recurrent weight matrix **W** _hh_, initial state **h** 0, and readout
weight matrix **W** _yh_ . Biases for latent and readout units are given by **b** _h_ and **b** _y_ .


Long Short Term Memory (LSTM) networks [16] are a more complex variant of RNNs that often
prove more powerful. In LSTMs latent units retain their values until explicitly cleared by the action
of a ‘forget gate’. They thus more naturally retain information for many time steps, which is believed
to make them easier to train. Additionally, hidden units are updated using multiplicative interactions,
and they can thus perform more complicated transformations for the same number of latent units.
The update equations for an LSTM are significantly more complicated than for an RNN, and can be
found in Appendix A.


**3.2** **Input and Output Time Series**


In order to train an RNN or LSTM on student interactions, it is necessary to convert those interactions into a sequence of fixed length input vectors **x** _t_ . We do this using two methods depending on
the nature of those interactions:


For datasets with a small number _M_ of unique exercises, we set _xt_ to be a one-hot encoding of
the student interaction tuple _ht_ = _{qt, at}_ that represents the combination of which exercise was
answered and if the exercise was answered correctly, so _xt ∈{_ 0 _,_ 1 _}_ [2] _[M]_ . We found that having
separate representations for _qt_ and _at_ degraded performance.


For large feature spaces, a one-hot encoding can quickly become impractically large. For datasets
with a large number of unique exercises, we therefore instead assign a random vector **n** _q,a ∼_
_N_ (0 _,_ **I** ) to each input tuple, where **n** _q,a ∈R_ _[N]_, and _N ≪_ _M_ . We then set each input vector
**x** _t_ to the corresponding random vector, **x** _t_ = **n** _qt,at_ . This random low-dimensional representation
of a one-hot high-dimensional vector is motivated by compressed sensing. Compressed sensing
states that a _k_ -sparse signal in _d_ dimensions can be recovered exactly from _k_ log _d_ random linear
projections (up to scaling and additive constants) [2]. Since a one-hot encoding is a 1-sparse signal,
the student interaction tuple can be exactly encoded by assigning it to a fixed random Gaussian input
vector of length _∼_ log 2 _M_ . Although the current paper deals only with 1-hot vectors, this technique
can be extended easily to capture aspects of more complex student interactions in a fixed length

vector.


The output **y** _t_ is a vector of length equal to the number of problems, where each entry represents
the predicted probability that the student would answer that particular problem correctly. Thus the
prediction of _at_ +1 can then be read from the entry in _yt_ corresponding to _qt_ +1.


**3.3** **Optimization**


The training objective is the negative log likelihood of the observed sequence of student responses
under the model. Let _δ_ ( _qt_ +1) be the one-hot encoding of which exercise is answered at time _t_ + 1,
and let _ℓ_ be binary cross entropy. The loss for a given prediction is _ℓ_ ( **y** _[T]_ _δ_ ( _qt_ +1) _, at_ +1), and the


4


loss for a single student is:



_L_ = � _ℓ_ ( **y** _[T]_ _δ_ ( _qt_ +1) _, at_ +1) (3)

_t_



This objective was minimized using stochastic gradient descent on minibatches. To prevent overfitting during training, dropout was applied to **h** _t_ when computing the readout **y** _t_, but not when
computing the next hidden state **h** _t_ +1. We prevent gradients from ‘exploding’ as we backpropagate
through time by truncating the length of gradients whose norm is above a threshold. For all models
in this paper we consistently used hidden dimensionality of 200 and a mini-batch size of 100. To
facilitate research in DKTs we have published our code and relevant preprocessed data [1] .


**4** **Educational Applications**


The training objective for knowledge tracing is to predict a student’s future performance based on
their past activity. This is directly useful – for instance formal testing is no longer necessary if a
student’s ability undergoes continuous assessment. As explored experimentally in Section 6, the
DKT model can also power a number of other advancements.


**4.1** **Improving Curricula**


One of the biggest potential impacts of our model is in choosing the best sequence of learning items
to present to a student. Given a student with an estimated hidden knowledge state, we can query
our RNN to calculate what their expected knowledge state would be if we were to assign them a
particular exercise. For instance, in Figure 1 after the student has answered 50 exercises we can test
every possible next exercise we could show her and compute her expected knowledge state given that
choice. The predicted optimal next problem for this student is to revisit solving for the y-intercept.


We use a trained DKT to test two classic curricula rules from education literature: _mixing_ where
exercises from different topics are intermixed, and _blocking_ where students answer series of exercises of the same type [30]. Since choosing the entire sequence of next exercises so as to maximize
predicted accuracy can be phrased as a Markov decision problem we can also evaluate the benefits
of using the expectimax algorithm (see Appendix) to chose an optimal sequence of problems.


**4.2** **Discovering Exercise Relationships**


The DKT model can further be applied to the task of discovering latent structure or concepts in the
data, a task that is typically performed by human experts. We approached this problem by assigning
an influence _Jij_ to every directed pair of exercises _i_ and _j_,


_y_ ( _j|i_ )
_Jij_ = ~~�~~ _k_ _[y]_ [ (] _[j][|][k]_ [)] _[,]_ (4)


where _y_ ( _j|i_ ) is the correctness probability assigned by the RNN to exercise _j_ on the second timestep,
given that a student answered exercise _i_ correctly on the first. We show that this characterization of
the dependencies captured by the RNN recovers the pre-requisites associated with exercises.


**5** **Datasets**


We test the ability to predict student performance on three datasets: simulated data, Khan Academy
Data, and the Assistments benchmark dataset. On each dataset we measure area under the curve
(AUC). For the non-simulated data we evaluate our results using 5-fold cross validation and in all
cases hyper-parameters are learned on training data. We compare the results of Deep Knowledge
Tracing to standard BKT and, when possible to optimal variations of BKT. Additionally we compare
our results to predictions made by simply calculating the marginal probability of a student getting a
particular exercise correct.


1https://github.com/chrispiech/DeepKnowledgeTracing


5


_Overview_ _AUC_


Dataset Students Exercise Tags Answers Marginal BKT BKT* DKT


Simulated-5 4,000 50 200 K ? 0.54 - 0.75

Khan Math 47,495 69 1,435 K 0.63 0.68 - 0.85

Assistments 15,931 124 526 K 0.62 0.67 0.69 0.86


_Table 1: AUC results for all datasets tested. BKT is the standard BKT. BKT* is the best reported result from_
_the literature for Assistments. DKT is the result of using LSTM Deep Knowledge Tracing._


**Simulated Data** : We simulate virtual students learning virtual concepts and test how well we can
predict responses in this controlled setting. For each run of this experiment we generate two thousand students who answer 50 exercises drawn from _k ∈_ 1 _. . ._ 5 concepts. For this dataset only, all
students answer the same sequence of 50 exercises. Each student has a latent knowledge state “skill”
for each concept, and each exercise has both a single concept and a difficulty. The probability of
a student getting a exercise with difficulty _β_ correct if the student had concept skill _α_ is modelled
using classic Item Response Theory [9] as: _p_ (correct _|α, β_ ) = _c_ + 1+1 _e−_ _[β]_ _c_ _[−][α]_ [where] _[ c]_ [ is the probabil-]
ity of a random guess (set to be 0.25). Students “learn” over time via an increase to the concept
skill which corresponded to the exercise they answered. To understand how the different models
can incorporate unlabelled data, we do _not_ provide models with the hidden concept labels (instead
the input is simply the exercise index and whether or not the exercise was answered correctly). We
evaluate prediction performance on an additional two thousand simulated test students. For each
number of concepts we repeat the experiment 20 times with different randomly generated data to
evaluate accuracy mean and standard error.


**Khan Academy Data** : We used a sample of anonymized student usage interactions from the eighth
grade Common Core curriculum on Khan Academy. The dataset included 1.4 million exercises
completed by 47,495 students across 69 different exercise types. It did not contain any personal
information. Only the researchers working on this paper had access to this anonymized dataset, and
its use was governed by an agreement designed to protect student privacy in accordance with Khan
Academy’s privacy notice [1]. Khan Academy provides a particularly relevant source of learning
data, since students often interact with the site for an extended period of time and for a variety of
content, and because students are often self-directed in the topics they work on and in the trajectory
they take through material.


**Benchmark Dataset** : In order to understand how our model compared to other models we evaluated
models on the Assistments 2009-2010 “skill builder” public benchmark dataset [2] . Assistments is an
online tutor that simultaneously teaches and assesses students in grade school mathematics. It is, to
the best of our knowledge, the largest publicly available knowledge tracing dataset [11].


**6** **Results**


On all three datasets Deep Knowledge Tracing substantially outperformed previous methods. On
the Khan dataset using an LSTM neural network model led to an AUC of 0.85 which was a notable
improvement over the performance of a standard BKT (AUC = 0.68), especially when compared to
the small improvement BKT provided over the marginal baseline (AUC = 0.63). See Table 1 and
Figure 3(b). On the Assistments dataset DKT produced a 25% gain over the previous best reported
result (AUC = 0.86 and 0.69 respectively) [23]. The gain we report in AUC compared to the marginal
baseline (0.24) is more than triple the largest gain achieved on the dataset to date (0.07).


The prediction results from the synthetic dataset provide an interesting demonstration of the capacities of deep knowledge tracing. Both the LSTM and RNN models did as well at predicting student
responses as an oracle which had perfect knowledge of all model parameters (and only had to fit the
latent student knowledge variables). See Figure 3(a). In order to get accuracy on par with an oracle
the models would have to mimic a function that incorporates: latent concepts, the difficulty of each
exercise, the prior distributions of student knowledge and the increase in concept skill that happened


2https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data


6


0.85


0.75


0.65


0.55





















answer questions and evaluate how much a student knew after 30 exercises. We repeated student
simulations 500 times and measured the average predicted probability of a student getting future
questions correct. In the Assistment context the blocking strategy had a notable advantage over
mixing. See Figure 3(c). While blocking performs on par with solving expectimax one exercise
deep (MDP-1), if we look further into the future when choosing the next problem we come up with
curricula where students have higher predicted knowledge after solving fewer problems (MDP-8).


The prediction accuracy on the synthetic dataset suggest that it may be possible to use DKT models
to extract the latent structure between the assessments in the dataset. The graph of our model’s
conditional influences for the synthetic dataset reveals a perfect clustering of the five latent concepts
(see Figure 4), with directed edges set using the influence function in Equation 4. An interesting
observation is that some of the exercises from the same concept occurred far apart in time. For
example, in the synthetic dataset, where node numbers depict sequence, the 5th exercise in the
synthetic dataset was from hidden concept 1 and even though it wasn’t until the 22nd problem
that another problem from the same concept was asked, we were able to learn a strong conditional
dependency between the two. We analyzed the Khan dataset using the same technique. The resulting
graph is a compelling articulation of how the concepts in the 8th grade Common Core are related
to each other (see Figure 4. Node numbers depict exercise tags). We restricted the analysis to
ordered pairs of exercises _{A, B}_ such that after _A_ appeared, _B_ appeared more than 1% of the
time in the remainder of the sequence). To determine if the resulting conditional relationships are a
product of obvious underlying trends in the data we compared our results to two baseline measures
(1) the transition probabilities of students answering _B_ given they had just answered _A_ and (2) the
probability in the dataset (without using a DKT model) of answering _B_ correctly given a student
had earlier answered _A_ correctly. Both baseline methods generated discordant graphs, which are
shown in the Appendix. While many of the relationships we uncovered may be unsurprising to an
education expert their discovery is affirmation that the DKT network learned a coherent model.


**7** **Discussion**


In this paper we apply RNNs to the problem of knowledge tracing in education, showing improvement over prior state-of-the-art performance on the Assistments benchmark and Khan dataset. Two
particularly interesting novel properties of our new model are that (1) it does not need expert annotations (it can learn concept patterns on its own) and (2) it can operate on any student input that can
be vectorized. One disadvantage of RNNs over simple hidden Markov methods is that they require
large amounts of training data, and so are well suited to an online education environment, but not a
small classroom environment.


7


6






























|5 2 462 86 13<br>3 0 3 3 1 41 13 24 61 3 7 732 394448 49 7 81 21 5 22 90 273 43 43464 47 0 1 15 92 14 2 12 89 1014 1621 23 34 23 8835 H H H H Hi i i i id d d d dd d d d de e e e en n n n c c c c co o o o on n n n nc c c c ce e e e ep p p p pt t t t 1 2 3 4<br>mulated Data 2 5 n t 5<br>h 3an Dat 6La<br>8n3 es9 6957 44 1S 3c na gtt le er s9 4pl 6ot 4s 111 192815 nction2 s4 62P 6y t 5t 5hh e 5a og ro e 4r me 8an 1 2 3 L R L li i qe on n sc pe e to ela a eig tmpr r on lf e i nu cq lz an ai u n ftc wa ig it eot ni oi qo noi rr un n dr a ns i pt n i3i rot o oce nn br sac ele mp n ifit u ss cm nb oe tr as 2 2 2 34 5 6 I S S An y o nt as le gu wt pr le tp nimr o ee ns 2ut si o nn pt rf cg o e lf sq eu pyu sn osa tc t rt ie tt i ii mo oo nn nn ss o g lw fr a . ee p E lq ah ul tis iam oti n. o s0 n hs 4 4 4 57 8 9 C P S Sy c cio ai itn be ech ls en a it t tsg or ri tu ofi fio neac or sc cpt e hi sn na en sn o oig t t a ei h aun t tpte i iic io oeto oo an nr nn te sis m nn i gs t ut p .e ir dtn eo i it o cso ins tf mry s is at le stm<br>4 i A 5652 Fu7 4 5 M P Sa u r ua l at l e ol si i n le s e2 i s i e ln t tion 2 2 7 8 V R xi e e c o egs o nf f a f 2u n c o 5 5 0 1 L M un u e t i g ir p q n u at i o in w b u ion<br>9 6 S y o a t 2 9 G r h i g r o a r ips 5 2 F r nt a tr d<br>7 E i s 3 0 E p o n t u 5 3 C t<br>8 f 1 s 4<br>3 72 5 31 49Exponent1 s4 1 19 10 L S Py li on s te t tea im nr gm s too hfd e e ll q is nu eo a f t ib o i n bv s ea swr ti a i fitt he ed la it ma ination 3 3 32 3 U E Sexn gpd moe nr es e nt na tn asd 2 de iq tiu oa ntions word problems 5 5 55 6 P F Vy u ert nh tc ia ctg i aoo ln r ase a 1 gn et sh e 2orem 2<br>66 33 e o f t 4 t d 7 n l<br>60 53 54 30 12 Integer sums 35 Systems of equations w. substitution 58 Solving for the x intercept<br>26 13 Congruent angles 36 Comparing proportional relationships 59 Recognizing functions<br>qs ute am tios o sf 43 17 42 gra5 ph8 61 42 67 4 1 1 1 14 5 6 E I R Lnx ie rt nap e p epro e ap han rr t e e i nnn t oi gt lns sg d 1 e is ncc eia tm ait ot rae flr us p nttl oo co tst f is or ya nsc seti mon ss 2 3 3 3 47 8 9 S F M Co i on il ld nu d up st i ti n o ro g ei un n cwis t tn iot ot no e f r dal ci n e ps cpe e aoa t g ts tbr m loe reef q pn u l lti sa ont tei so arn s functions 6 6 6 60 1 2 S S D Aq l oi go su ep ta a ve wr n e a c tnr e rio dd nfo go ptt r rs rdi m oa en bu cg lil emal me s slsim toil a frr ait cy<br>n 22 Line s 1 51 23 63 1 7 8 G i c a n o l l u n s t 4 0 1 V o m r g s r e m 6 3 4 C n e r o a tions 2<br>0 8 2 19 Interpreting features of linear functions 42 Solving for the y intercept 65 Pythagorean theorem 1<br>35 25 50 36 2938 3 37 16 F 2r 0actions 52 2 2 2 20 1 2 R C G Ce o orap n mpe s pha tr ut iui n tn c ig ng t i gl nd i ng ie ne c l a i i sm rn e e ia ea ql nrs u f at iuo fitin cof c r n nta si oc o tt n aio s tin os 1 4 4 4 43 4 5 G F C Ar or nea m gqp lu ph ee sain n r 1ig c ni e gsy s f s o et afe tm b ui rs v e ao sr f oi ae ftq e fu u da nat ci to ta in os ns 1 6 6 6 66 7 8 C O A Pao r n rdm g ae l lp r e ls a a lr o di lfn id ng m i et if sa oe g 1na nt pu it or ue sds te u o lf a tf eunctions 0<br>3 c t n 6 9 e|Col2|Col3|
|---|---|---|
|1<br>8<br>38<br>58<br>60<br>2<br>14<br>3<br>51<br>4<br>23<br>30<br>49<br>54<br>5<br>13<br>31<br>46<br>57<br>68<br>69<br>7<br>26<br>43<br>22<br>50<br>9<br>11<br>44<br>0<br>25<br>35<br>15<br>12<br>64<br>33<br>41<br>16<br>20<br>17<br>19<br>56<br>42<br>24<br>27<br>59<br>28<br>29<br>36<br>32<br>34<br>39<br>37<br>48<br>55<br>65<br>52<br>63<br>53<br>62<br>66<br>67<br>1 Linear function intercepts<br>24 Interpreting function graphs<br>47 Constructing inconsistent system<br>2 Recognizing irrational numbers<br>25 Systems of equations w. Elim. 0<br>48 Pythagorean theorem proofs<br>3 Linear equations 3<br>26 Solutions to systems of equations<br>49 Scientiﬁc notation intuition<br>4 Multiplication in scientiﬁc notation<br>27 Views of a function<br>50 Line graph intuition<br>5 Parallel lines 2<br>28 Recog func 2<br>51 Multistep equations w. distribution<br>6 Systems of equations<br>29 Graphing proportional relationships<br>52 Fractions as repeating decimals<br>7 Equations word problems<br>30 Exponent rules<br>53 Cube roots<br>8 Slope of a line<br>31 Angles 2<br>54 Scientiﬁc notation<br>9 Linear models of bivariate data<br>32 Understand equations word problems 55 Pythagorean theorem 2<br>10 Systems of equations with elimination<br>33 Exponents 2<br>56 Functions 1<br>11 Plotting the line of best ﬁt<br>34 Segment addition<br>57 Vertical angles 2<br>12 Integer sums<br>35 Systems of equations w. substitution<br>58 Solving for the x intercept<br>13 Congruent angles<br>36 Comparing proportional relationships 59 Recognizing functions<br>14 Exponents 1<br>37 Solutions to linear equations<br>60 Square roots<br>15 Interpreting scatter plots<br>38 Finding intercepts of linear functions 61 Slope and triangle similarity<br>16 Repeating decimals to fractions 2<br>39 Midpoint of a segment<br>62 Distance formula<br>17 Graphical solutions to systems<br>40 Volume word problems<br>63 Converting decimals to fractions 2<br>18 Linear non linear functions<br>41 Constructing scatter plots<br>64 Age word problems<br>19 Interpreting features of linear functions 42 Solving for the y intercept<br>65 Pythagorean theorem 1<br>20 Repeating decimals to fractions 1<br>43 Graphing systems of equations<br>66 Comparing features of functions 0<br>21 Constructing linear functions<br>44 Frequencies of bivariate data<br>67 Orders of magnitude<br>22 Graphing linear equations<br>45 Comparing features of functions 1<br>68 Angle addition postulate<br>23 Computing in scientiﬁc notation<br>46 Angles 1<br>69 Parallel lines 1<br>Scatter plots<br>Pythagorean<br>theorem<br>Line graphs<br>stems of<br>quations<br>Lines<br>Functions<br>Exponents<br>Fractions<br>Angles<br>12<br>24<br>19<br>40<br>46<br>47<br>15<br>5 22<br>30 31<br>36<br>41<br>42<br>6<br>7<br>8<br>3<br>18<br>11<br>20<br>25<br>27<br>29<br>33<br>43<br>45<br>1<br>2<br>8<br>9<br>10<br>14<br>16<br>23<br>28<br>34<br>35<br>38<br>21<br>4<br>2632<br>37<br>39<br>44<br>48<br>49<br>1317<br>mulated Data<br>han Data<br>Hidden concept 1<br>Hidden concept 2<br>Hidden concept 3<br>Hidden concept 4<br>Hidden concept 5|24 Interpreting function graphs<br>25 Systems of equations w. Elim. 0<br>26 Solutions to systems of equations<br>27 Views of a function<br>28 Recog func 2<br>29 Graphing proportional relationships<br>30 Exponent rules<br>31 Angles 2<br>32 Understand equations word problems<br>33 Exponents 2<br>34 Segment addition<br>35 Systems of equations w. substitution<br>36 Comparing proportional relationships<br>37 Solutions to linear equations<br>38 Finding intercepts of linear functions<br>39 Midpoint of a segment<br>40 Volume word problems<br>41 Constructing scatter plots<br>      42 Solving for the y intercept<br>43 Graphing systems of equations<br>44 Frequencies of bivariate data<br>45 Comparing features of functions 1<br>46 Angles 1|47 Constructing inconsistent system<br>48 Pythagorean theorem proofs<br>49 Scientiﬁc notation intuition<br>50 Line graph intuition<br>51 Multistep equations w. distribution<br>52 Fractions as repeating decimals<br>53 Cube roots<br>54 Scientiﬁc notation<br>     55 Pythagorean theorem 2<br>56 Functions 1<br>57 Vertical angles 2<br>58 Solving for the x intercept<br>    59 Recognizing functions<br>60 Square roots<br>      61 Slope and triangle similarity<br>62 Distance formula<br>63 Converting decimals to fractions 2<br>64 Age word problems<br>65 Pythagorean theorem 1<br>66 Comparing features of functions 0<br>67 Orders of magnitude<br>68 Angle addition postulate<br>69 Parallel lines 1|



_Figure 4: Graphs of conditional influence between exercises in DKT models. Above: We observe a perfect_
_clustering of latent concepts in the synthetic data. Below: A convincing depiction of how 8th grade math_
_Common Core exercises influence one another. Arrow size indicates connection strength. Note that nodes may_
_be connected in both directions. Edges with a magnitude smaller than 0.1 have been thresholded. Cluster_
_labels are added by hand, but are fully consistent with the exercises in each cluster._


The application of RNNs to knowledge tracing provides many directions for future research. Further investigations could incorporate other features as inputs (such as time taken), explore other
educational impacts (such as hint generation, dropout prediction), and validate hypotheses posed in
education literature (such as spaced repetition, modeling how students forget). Because DKTs take
vector input it should be possible to track knowledge over more complex learning activities. An especially interesting extension is to trace student knowledge as they solve open-ended programming
tasks [26, 27]. Using a recently developed method for vectorization of programs [25] we hope to be
able to intelligently model student knowledge over time as they learn to program.


In an ongoing collaboration with Khan Academy, we plan to test the efficacy of DKT for curriculum
planning in a controlled experiment, by using it to propose exercises on the site.


**Acknowledgments**


Many thanks to John Mitchell for his guidance and Khan Academy for its support. Chris Piech is
supported by NSF-GRFP grant number DGE-114747.


**References**


[1] Khan academy privacy notice https://www.khanacademy.org/about/privacy-policy, 2015.


[2] BARANIUK, R. Compressive sensing. _IEEE signal processing magazine 24_, 4 (2007).


[3] CEN, H., KOEDINGER, K., AND JUNKER, B. Learning factors analysis–a general method for cognitive
model evaluation and improvement. In _Intelligent tutoring systems_ (2006), Springer, pp. 164–175.


[4] COHEN, G. L., AND GARCIA, J. Identity, belonging, and achievement a model, interventions, implications. _Current Directions in Psychological Science 17_, 6 (2008), 365–369.


[5] CORBETT, A. Cognitive computer tutors: Solving the two-sigma problem. In _User Modeling 2001_ .
Springer, 2001, pp. 137–147.


[6] CORBETT, A. T., AND ANDERSON, J. R. Knowledge tracing: Modeling the acquisition of procedural
knowledge. _User modeling and user-adapted interaction 4_, 4 (1994), 253–278.


[7] D BAKER, R. S. J., CORBETT, A. T., AND ALEVEN, V. More accurate student modeling through
contextual estimation of slip and guess probabilities in bayesian knowledge tracing. In _Intelligent Tutoring_
_Systems_ (2008), Springer, pp. 406–415.


8


[8] D BAKER, R. S. J., PARDOS, Z. A., GOWDA, S. M., NOORAEI, B. B., AND HEFFERNAN, N. T.
Ensembling predictions of student knowledge within intelligent tutoring systems. In _User Modeling,_
_Adaption and Personalization_ . Springer, 2011, pp. 13–24.

[9] DRASGOW, F., AND HULIN, C. L. Item response theory. _Handbook of industrial and organizational_
_psychology 1_ (1990), 577–636.

[10] ELLIOT, A. J., AND DWECK, C. S. _Handbook of competence and motivation_ . Guilford Publications,
2013.

[11] FENG, M., HEFFERNAN, N., AND KOEDINGER, K. Addressing the assessment challenge with an online
system that tutors as it assesses. _User Modeling and User-Adapted Interaction 19_, 3 (2009), 243–266.

[12] FITCH, W. T., HAUSER, M. D., AND CHOMSKY, N. The evolution of the language faculty: clarifications
and implications. _Cognition 97_, 2 (2005), 179–210.

[13] GENTNER, D. Structure-mapping: A theoretical framework for analogy. _Cognitive science 7_, 2.

[14] GONG, Y., BECK, J. E., AND HEFFERNAN, N. T. In _Intelligent Tutoring Systems_, Springer.

[15] GRAVES, A., MOHAMED, A.-R., AND HINTON, G. Speech recognition with deep recurrent neural
networks. In _Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference_
_on_ (2013), IEEE, pp. 6645–6649.

[16] HOCHREITER, S., AND SCHMIDHUBER, J. Long short-term memory. _Neural computation 9_, 8.

[17] KARPATHY, A., AND FEI-FEI, L. Deep visual-semantic alignments for generating image descriptions.
_arXiv preprint arXiv:1412.2306_ (2014).

[18] KHAJAH, M., WING, R. M., LINDSEY, R. V., AND MOZER, M. C. Incorporating latent factors into
knowledge tracing to predict individual differences in learning. _Proceedings of the 7th International_
_Conference on Educational Data Mining_ (2014).

[19] KHAJAH, M. M., HUANG, Y., GONZ ´ALEZ-BRENES, J. P., MOZER, M. C., AND BRUSILOVSKY, P.
Integrating knowledge tracing and item response theory: A tale of two frameworks. _Proceedings of the_
_4th International Workshop on Personalization Approaches in Learning Environments_ (2014).

[20] LAN, A. S., STUDER, C., AND BARANIUK, R. G. Time-varying learning and content analytics via
sparse factor analysis. In _Proceedings of the 20th ACM SIGKDD international conference on Knowledge_
_discovery and data mining_ (2014), ACM, pp. 452–461.

[21] LINNENBRINK, E. A., AND PINTRICH, P. R. Role of affect in cognitive processing in academic contexts.
_Motivation, emotion, and cognition: Integrative perspectives on intellectual functioning and development_
(2004), 57–87.

[22] MIKOLOV, T., KARAFI ´AT, M., BURGET, L., CERNOCK `Y, J., AND KHUDANPUR, S. Recurrent neural
network based language model. In _INTERSPEECH 2010, 11th Annual Conference of the International_
_Speech Communication Association, Makuhari, Chiba, Japan, 2010_ (2010), pp. 1045–1048.

[23] PARDOS, Z. A., AND HEFFERNAN, N. T. Kt-idem: Introducing item difficulty to the knowledge tracing
model. In _User Modeling, Adaption and Personalization_ . Springer, 2011, pp. 243–254.

[24] PAVLIK JR, P. I., CEN, H., AND KOEDINGER, K. R. Performance Factors Analysis–A New Alternative
to Knowledge Tracing. _Online Submission_ (2009).

[25] PIECH, C., HUANG, J., NGUYEN, A., PHULSUKSOMBATI, M., SAHAMI, M., AND GUIBAS, L. J.
Learning program embeddings to propagate feedback on student code. _CoRR abs/1505.05969_ (2015).

[26] PIECH, C., SAHAMI, M., HUANG, J., AND GUIBAS, L. Autonomously generating hints by inferring
problem solving policies. In _Proceedings of the Second (2015) ACM Conference on Learning @ Scale_
(New York, NY, USA, 2015), L@S ’15, ACM, pp. 195–204.

[27] PIECH, C., SAHAMI, M., KOLLER, D., COOPER, S., AND BLIKSTEIN, P. Modeling how students learn
to program. In _Proceedings of the 43rd ACM symposium on Computer Science Education_ .

[28] POLSON, M. C., AND RICHARDSON, J. J. _Foundations of intelligent tutoring systems_ . Psychology
Press, 2013.

[29] RAFFERTY, A. N., BRUNSKILL, E., GRIFFITHS, T. L., AND SHAFTO, P. Faster teaching by POMDP
planning. In _Artificial intelligence in education_ (2011), Springer, pp. 280–287.

[30] ROHRER, D. The effects of spacing and mixing practice problems. _Journal for Research in Mathematics_
_Education_ (2009), 4–17.

[31] SCHRAAGEN, J. M., CHIPMAN, S. F., AND SHALIN, V. L. _Cognitive task analysis_ . Psychology Press,
2000.

[32] WILLIAMS, R. J., AND ZIPSER, D. A learning algorithm for continually running fully recurrent neural
networks. _Neural computation 1_, 2 (1989), 270–280.

[33] YUDELSON, M. V., KOEDINGER, K. R., AND GORDON, G. J. Individualized bayesian knowledge
tracing models. In _Artificial Intelligence in Education_ (2013), Springer, pp. 171–180.


9


