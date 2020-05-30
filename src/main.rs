use std::collections::{HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// This function create a new dictionary file by taking original file and split
/// all dictionary entry if it contain whitespace into separate entry.
/// It using HashSet to store a dictionary entry to prevent duplication before flusing it
/// to target file.
fn clean_dict<P: AsRef<std::path::Path>>(source: P, target: P) -> std::io::Result<HashSet<String>> {
    let reader = BufReader::new(File::open(source)?);

    let mut dict = HashSet::new();

    reader.lines().for_each(|line| {
        line.unwrap().split_whitespace().for_each(|word| {
            dict.insert(word.to_owned());
        });
    });
    let mut writer = BufWriter::new(File::create(target)?);
    dict.iter().for_each(|word| {
        writer.write(word.as_bytes()).unwrap();
        writer.write(b"\n").unwrap();
    });
    Ok(dict)
}

fn main() {
    use rand::seq::SliceRandom;
    use std::time::{Instant};
    use tokenizer::Tokenizer;
    use tokenizer::th;

    let original_dict_path = "data/lexitron_utf8.txt";
    let clean_dict_path = "data/lexitron_mod.txt";
    let dict = clean_dict(original_dict_path, clean_dict_path).unwrap();
    let words: Vec<String> = dict.into_iter().collect(); // turn hashset into vec
    let mut predicted_positive = 0;
    let mut actual_positive = 0;
    let mut true_positive = 0;
    let begin = Instant::now();
    let mut cumulative_f1 = 0f64;
    let montecarlo_times = 10;
    let sampling_size = 200;
    let validation_ratio = 0.1; // 10% of every test is unknown word.
    let mut rng = rand::thread_rng();

    for sim_idx in 0..montecarlo_times {
        let sample: Vec<&String> = words.choose_multiple(&mut rng, sampling_size).collect();
        let split_point = (sampling_size as f64 * validation_ratio) as usize;
        let words = &sample[split_point..]; // Only add portion of words to dict to see how it handle unknown word
        let instantiate_time = Instant::now();
        let tokenizer = th::Tokenizer::from(words);
        println!("Simulation {} has total tokenizer instantiate time {} ms", sim_idx, instantiate_time.elapsed().as_millis());

        // trigram word validation
        permutator::k_permutation(&sample, 3, |product| {
            let combined = format!("{}{}{}", product[0], product[1], product[2]);
            let tokens = tokenizer.tokenize(&combined);
            actual_positive += 3;
            predicted_positive += tokens.len();

            let mut i = 0;
            let mut j = 0;

            while i < tokens.len() {
                let mut cum_len = 0;
                while i < tokens.len() && cum_len < product[j].len() {
                    // potential too much tokenize token[0]
                    cum_len += tokens[i].len();
                    i += 1;
                }

                if tokens[i - 1] == product[j].as_str() {
                    true_positive += 1;
                }

                j += 1;
            }
        });

        let processed_time = begin.elapsed().as_millis();
        let precision = (true_positive as f64) / (predicted_positive as f64);
        let recall = (true_positive as f64) / (actual_positive as f64);
        let f1_score = 2f64 * (precision * recall) / (precision + recall);
        cumulative_f1 += f1_score;

        println!("Simulation {} got F1 score = {}", sim_idx, f1_score);
        println!("Simulation {} take total processing time = {} m {} s {} ms", sim_idx, processed_time / 60_000, (processed_time / 1000) % 60, processed_time % 1000);
    }

    println!("Average F1 score = {}", cumulative_f1 / montecarlo_times as f64);
}