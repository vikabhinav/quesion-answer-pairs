Problem Definition 

There are organization which are exploring to control the expense that are recurring and simple to medium complex in nature. In this highly competitive technology niche, where the customer is always on the hunt for the more customized solution, it is tough to satisfy the requirement of customers. Users and customers are getting smarter and will likely to move towards the solution that is qualitative and at the same time simple to use. In this scenario, chatbots comes as an easy solution to satisfy the need of today’s customer base. 
We are targeting to explore a solution that is simple and can be quickly adopted in the real life and can be plugged in any solution where question are asked by a requester (or customer or human being) and the response would be provided by a chatbot.

Literature Survey
During this period the focus is the identify the right set of data that would fit in the business problem.
The generator prepares the feature-rich input embedding — a concatenation of (a) a refined coreference position feature embedding, (b) an answer feature embedding and (c) a word embedding, each of which is described below. It then encodes the textual input using an LSTM unit (Hochreiter and Schmidhuber,1997). Finally an attention-copy equipped decoder is used to decode the question. More specifically, given the input sentence S (containing an answer span) and the preceding context C, we first run a coreference resolution system to get the coref-clusters for S and C and use them to create a coreference transformed input sentence: for each pronoun, we append its most representative non-pronominal coreferentmention. Specifically, we apply the simple feed forward network-based mention-ranking model of Clark and Manning (2016) to the concatenation of C and S to get the coref-clusters for all entities in C and S. The C&M model produces a score/representation s for each mention pair (m1,m2),
s(m1;m2)=Wmhm(m,m2)+ bm      (2)

where W_m is a 1 * d weight matrix and b is the bias. h_m (m1, m2) is representation of the last hidden layer of the three-layer feed forward neural network. For each pronoun in S, we then heuristically identify the most “representative” antecedent from
its coref-cluster. (Proper nouns are preferred.) We append the new mention after the pronoun. For example, in Table 1, “the panthers” is the most representative mention in the coref-cluster for “they”. The new sentence with the appended coreferentmention is our coreference transformed input sentenceS^' (see Figure 2).  Coreference Position Feature Embedding For
each token in S^' , we also maintain one position featuref^c=(C_1,…,C_n). to denote pronouns (e.g.,“they”) and antecedents (e.g., “the panthers”). We
use the BIO tagging scheme to label the associated spans in S^' . “B_ANT” denotes the start of anantecedent span, tag “I_ANT” continues the antecedent span and tag “O” marks tokens that do
not form part of a mention span. Similarly, tags “B_PRO” and “I_PRO” denote the pronoun span.

Inspired by the success of gating mechanisms for controlling information flow in neural networks (Hochreiter and Schmidhuber, 1997;Dauphin et al., 2017), we propose to use a gating network here to obtain a refined representation of the coreference position feature vectorsf^c=(C_1,…,C_n).. The main idea is to utilize the mention-pair score (see Equation 2) to help the neural network learn the importance of the coreferent phrases. We compute the refined (gated) coreference position feature vectorf^d=(d_1,…,d_n) as follows,
g_i= ReLU(W_a C_i+ W_b 〖score〗_i+b)
d_i= g_i ʘc_i
Where ʘ denotes an element-wise product between two vectors and ReLU is the rectified linear activation function. 〖score〗_i denotes the mention-pair score for each antecedent token (e.g., “the” and “panthers”) with the pronoun (e.g.,“they”);〖score〗_iis obtained from the trained model Equation2) of the C&M. If token i is not added later as an antecedent token, 〖score〗_iis set to zero. W_a,W_b are weight matrices and b is the bias Vector.
Answer Feature Embedding We also include an answer position feature embedding to generate answer-specific questions; we denote the answer span with the usual BIO tagging scheme (see, e.g., “the arizona cardinals” in Table 1 or “O”) is mapped to its feembedding space: f^a=(a_1,…,a_n).



Word Embedding To obtain the word embedding for the tokens themselves, we just map the tokens to the word embedding space:x =(x_1,…,x_n).
Final Encoder Input As noted above, the final input to the LSTM-based encoder is a concatenation of (1) the refined coreference position feature embedding (light blue units in Figure 2), (2)the answer position feature embedding (red units),and (3) the word embedding for the token (greenunits),
e_i=concat (d_i,a_i,x_i )               (4)
Encoder: As for the encoder itself, we use bi directional LSTMs to read the input e =(e_1,…,e_n)in both the forward and backward directions. After encoding, we obtain two sequences of hidden vectors, namely,
h ⃗=((h_1 ) ⃗,…,(h_n ) ⃗ )and h ⃖=((h_1 ) ⃖,…,(h_n ) ⃖).
The final output state of the encoderis the concatenation ofh ⃗  and h ⃖ where,
h_i=concat ((h_i ) ⃗,(h_i ) ⃖ )              

Sample Data
The data that has been chosen to review is from Stanford Question Answering Dataset (SQuAD) which is a reading comprehension dataset, consisting of questions posed by crowd workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarial by crowd workers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions, when possible, but also determine when no answer is supported by the paragraph and abstain from answering.
SQuAD2.0 tests the ability of a system to not only answer reading comprehension questions, but also abstain when presented with a question that cannot be answered based on the provided paragraph.
https://rajpurkar.github.io/SQuAD-explorer/

Tentative list of Algorithms

We are looking to explore the Natural Language processing to create a chatbot that can answer customer’s question. Natural languages are those that have naturally evolved over time by humans speaking and repeating it for communication with each other over the years, eg. English, French, Spanish, Hindi, etc. It is an interdisciplinary domain merging the fields of linguistics, computer science and artificial intelligence. NLP has two parts to it namely; Natural Language Understanding (NLU) and Natural Language Generation (NLG). NLU is the reading and understanding of the natural language that is generated by humans, by the machine, eg. Is this review positive or negative, what do people feel about a product, what is this tweet about, what does this user want from this particular query etc. NLG is the generation of natural language by the machine in response to human generated input eg. response by a chatbot, weather forecasts from data, automatic subtitle generation etc.
By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation.
