# merkle-tree-models
Using zero knowledge proofs to prove model history/evolution in a tree

## Idea
Sometimes, people train machine learning models that are derived from prior trained machine learning models. It would be good to have a way to prove full history of such models where each entity training the models gets credit for them having trained or finetuned a model, possibly on their own private data.

The ideal situation is that Alice trains a model and releases the weights along with a proof that she trained them from a particular random initialization. Bob then uses Alice's model and trains them on his data, releasing the weights and provides a proof he trained it from Alice's trained weights. We can then prove that Bob's model eventually came from the random initialization that Alice used. This could be interesting for a case where a single foundation model is finetuned many times, branching out for many different tasks.

## Related work
[1] https://arxiv.org/abs/2307.16273 -- Zero knowledge proofs of deep learning **training**. Supports only "arithmetic operations" -- in this case, ReLU activation and MSE loss. Undocumented code is https://github.com/jvhs0706/zkdl-train/ but documented code (for **inference** only) is https://github.com/SafeAILab/zkDL .

[2] https://eprint.iacr.org/2024/162.pdf -- Zero knowledge proofs of **training** (zkPoT) for deep neural networks. Code is here: https://github.com/zkPoTs/kaizen

[3] https://eprint.iacr.org/2023/1345 -- Zero knowledge proofs of **training** zkPoT for logistic regression models. Code is here: https://github.com/guruvamsi-policharla/zkpot

[4] https://arxiv.org/abs/2404.16109 -- Zero knowledge proofs of LLM **inference**.

[5] https://eprint.iacr.org/2024/703.pdf -- Zero knowledge proofs of deep learning **inference**, supporting softmax and other activations

[6] https://ezkl.xyz/ -- Library for zero knowledge proofs of machine learning model **inference**

## What may be possible given just these
Using the method described in [1], we might create a system where a simple ReLU network R on a regression task is trained on some data X_1, producing network weights R_1. We could generate the zk proof for R_init -> R_1, and then afterward that ReLU network R_1 can be trained separately on a regresssion task with other data X_2 and X_3, resulting in R_1,2 and R_1,3 respectively, and generate zk proofs for those runs. There are several ways to combine zk proofs in general, but I'm unsure of which ways are applicable in this situation of combining the proofs of R_1,2 and R_1 to prove the full model history. Even without any additional complex zero knowledge proof operations, we can just prove sequentially to show that a given model is part of a tree of models.

## In progress:
Provable evolutionary model merging. Model merging using evolutionary processes has been shown to increase capabilities of language models beyond their component models [7]. Our goal is to prove a merged model has a history of being created from parent models, enabling credit assignment of "who found what model", as well as proving what performance that model has on a particular task. We can think of this as a kind of decentralized model-merging leaderboard system. We are exploring using ezkl for this.

[7] https://arxiv.org/abs/2403.13187
