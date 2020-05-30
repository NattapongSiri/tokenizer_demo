# tokenizer_demo
A demo on how to use crate tokenizer to tokenize Thai text.

## Background
Thai language is a language used in Thailand. It is one of "Complex Text Language".
What make it complex is that it has no space between words. It has no clear rule on word boundary. It use various non-spacing mark characters. Name of entity usually comprise of regular Thai word. It is recommended to have space between each sentence but this rule is not strictly enforced. Thai language usually come mixing with borrowed word from foreign languages. Sometime these words come in romanization form. Despite the fact that Royal Institute of Thailand has a clear define rule for how to romanize Thai text, it is strictly adopt only in formal communication area such as News or academic research area.

## How to identify Thai word boundary ?
This is an actively research area for many decades. The early ear, researcher found that using Dictionary yield acceptable word break result. Later on, they start using other feature such as part-of-speech, rule-based, etc. It improve some specific case with specific constraints, e.g. no unknown word in text, no word ambiguity. In recent year, many research use deep learning to break Thai word. It show promising result with new kind of restriction, e.g. different word came out from different sentence even if it is exactly the same word. This is due to side effect of context around such word while training which may dominance the expected result. In practical use, Thai people use hybrid approach. Dictionary is still an essential part on every hybrid approach. This is because dictionary based is easier to maintain. It is easier to let user define new word.

## Word tokenizer in Rust
This repository demonstrate how to use `tokenizer` crate to tokenize Thai word by using Lexitron dictionary.
Lexitron is a dictionary built from NECTEC. See `data/lexitron-license` for licensing info of the dictionary.

See `src/main.rs` for complete code.
In there, there's two functions. `clean_dict` and `main`.

`clean_dict` function will break an entry that have space inside the entry into multiple entries. This function is needed because Lexitron dict entry contains multiple words per entry. 

`main` is where the program  entry is.
Let break down each chunk of code here
```rust
    use rand::seq::SliceRandom;
    use std::time::{Instant};
    use tokenizer::Tokenizer;
    use tokenizer::th;
```
These are used to import modules into Rust.
```rust
    let original_dict_path = "data/lexitron_utf8.txt";
    let clean_dict_path = "data/lexitron_mod.txt";
    let dict = clean_dict(original_dict_path, clean_dict_path).unwrap();
    let words: Vec<String> = dict.into_iter().collect(); // turn hashset into vec
```
These lines above clean dictionary. `clean_dict` return `HashSet` thus we need to convert it back to `Vec`
```rust
    let mut cumulative_f1 = 0f64;
    let montecarlo_times = 10;
    let sampling_size = 200;
    let validation_ratio = 0.1; // 10% of every test is unknown word.
```
Define metric and hyper-parameters to evaluate the tokenizer
```rust
    let mut rng = rand::thread_rng();
```
Use crate `rand` to get `random number generator`.
```rust
    let mut predicted_positive = 0;
    let mut actual_positive = 0;
    let mut true_positive = 0;
```
Now in each loop, we defined metric variables.
```rust
    let sample: Vec<&String> = words.choose_multiple(&mut rng, sampling_size).collect();
```
We use method from `SliceRandom` trait, `choose_multiple` to randomly pick some word from dict.
```rust
    let split_point = (sampling_size as f64 * validation_ratio) as usize;
    let words = &sample[split_point..]; // Only add portion of words to dict to see how it handle unknown word
```
We split sample into two set. Most of it will be used as dictionary for tokenizer.
```rust
    let instantiate_time = Instant::now();
    let tokenizer = th::Tokenizer::from(words);
    println!("Simulation {} has total tokenizer instantiate time {} ms", sim_idx, instantiate_time.elapsed().as_millis());
```
We obtains new tokenizer from given words slice dictionary. We start monitor a time it take to instantiate dictionary.
`Simulation 0 has total tokenizer instantiate time 0 ms`
It print following line in console. It mean it take less than 1 ms to instantiate tokenizer.
```rust
    let begin = Instant::now();

    permutator::k_permutation(&sample, 3, |product| {
        let combined = format!("{}{}{}", product[0], product[1], product[2]);
        let mut byte_count = 0;
        let expected : Vec<&str> = product.iter().map(|p| {
            let slice = &combined[byte_count..(byte_count + p.len())];
            byte_count += p.len();
            slice
        }).collect();
        // more code in here
    })
```
Above code take a start time to monitor how long it take to tokenize all these words. It use crate `permutator` to perform "k pick n" permutation. In other word, it construct all possible tri-gram of word from entire dictionary. There'll be some word that is not known by `Tokenizer`. We construct `expected` slice so that we can use memory trick to compare slice instead of comparing entire string to string.
```rust
    let tokens = tokenizer.tokenize(&combined);
    
    actual_positive += 3;
    predicted_positive += tokens.len();
```
Code above tokenize Thai words. Since we use tri-gram, actual positive will always increment by 3 on each test case. `predicted_positive` is equaals to number of tokens that `tokenizer` return.
```rust
    let mut i = 0;
    let mut j = 0;

    while i < tokens.len() {
        let mut cum_len = 0;
        while i < tokens.len() && cum_len < product[j].len() {
            // potential too much tokenize token[0]
            cum_len += tokens[i].len();
            i += 1;
        }

        if std::ptr::eq(tokens[i - 1], expected[j]) { 
            // compare by slice attribute rather than entire str comparison
            true_positive += 1;
        }

        j += 1;
    }
```
Above code compare each tokens to expected token by checking on pointer and length of it. It need to take into account when there's some fault break position which might or might not effect subsequence tokens. In case of fault token that is shorter than expected token, it need to consume some more tokens before continue comparing next token to next expectation.
```rust
    let processed_time = begin.elapsed().as_millis();
    let precision = (true_positive as f64) / (predicted_positive as f64);
    let recall = (true_positive as f64) / (actual_positive as f64);
    
    let f1_score = 2f64 * (precision * recall) / (precision + recall);
    cumulative_f1 += f1_score;
```
Above code calculate all the metrics. 
```rust
    println!("Simulation {} got F1 score = {}", sim_idx, f1_score);
    println!("Simulation {} take total processing time = {} m {} s {} ms", sim_idx, processed_time / 60_000, (processed_time / 1000) % 60, processed_time % 1000);
```
Above code print some of matrics.
```
Simulation 0 got F1 score = 0.9878095354485877
Simulation 0 take total processing time = 0 m 56 s 238 ms
```
Above are sample of metric printed by the code.
```rust
    println!("Average F1 score = {}", cumulative_f1 / montecarlo_times as f64);
```
Finally, it print average F1 score on entire experiment.
Here is an example of what it print.
`Average F1 score = 0.9851226929069391`

## Hyper-parameters choice
In here, we take 200 words sample from Lexitron dictionary.
We take 90% of sample and construct tokenizer out of it.
We take all 200 words and construct tri-gram of text.
Total possible case that got test is 200!/(3!(200-3)!) which is equals to 1,313,400 cases.
It take about 55s to complete 1,313,400 cases.
We choose to re-run everything for 10 times to see average F1 accuracy.