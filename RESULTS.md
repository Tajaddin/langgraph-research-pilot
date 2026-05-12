# HotpotQA benchmark results
Backend: `AnthropicClient` | Model: `claude-haiku-4-5-20251001` | n=30 | seed=7
## Headline
| Metric | Single-prompt baseline | Multi-step agent | Lift |
|---|---:|---:|---:|
| Exact match | 23.3% | 23.3% | +0.0% |
| F1 (token overlap) | 0.313 | 0.367 | +0.054 |
| Contains gold (substring) | 30.0% | 30.0% | +0.0% |
## Timing
- Baseline: total 24.1s, avg 0.8s/question
- Agent: total 535.9s, avg 17.86s/question

## Per-question detail
| # | Type | Level | Question | Gold | Baseline | Agent | Base F1 | Agent F1 |
|---|---|---|---|---|---|---|---:|---:|
| 1 | bridge | hard | Which of the city at the south of Shelly Beach, New South Wales is a local gover | City of Newcastle | Maitland | Lake Macquarie  Lake Macquarie is a local government area in | 0.00 | 0.00 |
| 2 | bridge | hard | Lee Sung-jae stars in the film Barking Dogs Never Bite which is what film genre? | comedy-drama | Dark comedy thriller | Dark comedy  (Barking Dogs Never Bite) | 0.00 | 0.00 |
| 3 | comparison | hard | What is name of the city in Istanbul which has both Column of the Goths and Gül  | Turkey | Fatih | Fatih  The Column of the Goths is located in Gülhane Park wi | 0.00 | 0.00 |
| 4 | bridge | hard | What was the manager for the 1891 Cincinnati Kelly's Killers also commonly known | $10,000 Kelly | Kelly's Killers manager. | King Kelly  King Kelly was the manager of the 1891 Cincinnat | 0.00 | 0.50 |
| 5 | comparison | hard | Which band was active the longest, Spirit of the West or Big Pig? | Spirit of the West | Spirit of the West | Spirit of the West  (Spirit of the West was active for 33 ye | 1.00 | 1.00 |
| 6 | comparison | hard | Which formed first, Fireflight or Dirty Three? | Dirty Three | Dirty Three | Dirty Three  Dirty Three was formed in 1992, while Firefligh | 1.00 | 1.00 |
| 7 | comparison | hard | What do Káťa Kabanovát and Der ferne Klang have in common? | an opera | Operas by Janáček and Schreker. | Struggle between sacred and profane love, redemption through | 0.00 | 0.00 |
| 8 | bridge | hard | The Little River Railroad is operated between Maryville and what pioneer Appalac | Elkmont, Tennessee | Gatlinburg | Elkmont  The Little River Railroad is operated between Maryv | 0.00 | 0.67 |
| 9 | bridge | hard | Which of the people featured on Wall of Fame is the daughter of Bernie Eccleston | Tamara Ecclestone | Tamara Ecclestone | Tamara Ecclestone  Tamara Ecclestone is identified as a daug | 1.00 | 1.00 |
| 10 | bridge | hard | Where was the NBA Draft held in which Harrison Bryce Jordan Barnes was selected  | Prudential Center in Newark | Barclays Center, Brooklyn. | Prudential Center in Newark, New Jersey  The 2012 NBA Draft, | 0.29 | 0.80 |
| 11 | bridge | hard | What vocalist and pianist was recorded by Modular Recordings? | Karen O | Arca | I cannot definitively answer this question based on the prov | 0.00 | 0.00 |
| 12 | bridge | hard | What year was the company, who released Forever Kingdom for PlayStation 2, found | 1986 | Forever Kingdom was released by Kei Corporation in 2001. Kei | 1997  FromSoftware released Forever Kingdom for PlayStation  | 0.00 | 0.00 |
| 13 | comparison | hard | Which aired first, Life After People or Werner Herzog Eats His Shoe? | Werner Herzog Eats His Shoe | Life After People. | Werner Herzog Eats His Shoe  Werner Herzog Eats His Shoe pre | 0.00 | 1.00 |
| 14 | bridge | hard | Jullian Allen a Brittish-American Illustrator had some of his Illustration publi | Harper's Bazaar | Harper's Bazaar | Harper's Bazaar UK  The publication formerly known as Harper | 1.00 | 0.80 |
| 15 | bridge | hard | Where is the company owning Reliance Cricket Stadium ranked on the Fortune Globa | 215th | Reliance Industries, 2016 rank: 155 | unknown | 0.00 | 0.00 |
| 16 | bridge | hard | The 34th Pursuit Squadron of the United States Air Force was among the 60,000-80 | World War II | World War II | World War II  The 34th Pursuit Squadron survivors were among | 1.00 | 1.00 |
| 17 | bridge | hard | Where is the head quarter of the Blue Bird Body Company? | Fort Valley, Georgia | Fort Valley, Georgia | Fort Valley, Georgia  Blue Bird Corporation is an American b | 1.00 | 1.00 |
| 18 | bridge | hard | What episode of South Park was the episode that aired in May before Funnybot? | 146th | "Assange" (Season 15, Episode 10) | unknown | 0.00 | 0.00 |
| 19 | bridge | hard | Shaqtin' a Fool has included which television personality who worked for 8 years | Matthew Ward Winer | Jalen Rose | unknown | 0.00 | 0.00 |
| 20 | bridge | hard | Which British engineering company is best known for its luxury motor cars in the | D. Napier & Son | Rolls-Royce | Rolls-Royce  Rolls-Royce became the most famous and prestigi | 0.00 | 0.00 |
| 21 | bridge | hard | Meomi Design works include The Octonauts book series, how did the adaptations of | a British children's television series | Netflix animated series adaptation. | Animated television series and streaming adaptations  (Sony  | 0.25 | 0.40 |
| 22 | bridge | hard | Minneapolis hip hop collective member that released album in 2006? | P.O.S | Atmosphere | unknown | 0.00 | 0.00 |
| 23 | bridge | hard | Renamed in 2014, what was the vehicle offered as a prize to contestants on the f | Chevrolet Corvette Stingrays | Toyota Corolla | unknown | 0.00 | 0.00 |
| 24 | bridge | hard | In what country did Irvine Gerald Sellar develop the second-tallest free-standin | United Kingdom | United States | unknown | 0.50 | 0.00 |
| 25 | comparison | hard | Are Benjamin Britten and Béla Bartók from the same country? | no | No, different countries. | No  England and Hungary are different countries; Benjamin Br | 0.50 | 1.00 |
| 26 | bridge | hard | What was the first air date that the CGI animated series that preceded "Mixels"? | January 16, 2013 | I need to find what CGI animated series preceded "Mixels." | I cannot determine the answer from the provided findings.  T | 0.00 | 0.00 |
| 27 | bridge | hard | Eduard Schweizer teaches at a German university with over how many students?  | 26,000 | I don't have specific information about Eduard Schweizer's u | I cannot answer this question as posed.  The findings clearl | 0.00 | 0.00 |
| 28 | bridge | hard | In which year was the King who made the 1925 Birthday Honours born? | 1865 | 1865 | unknown | 1.00 | 0.00 |
| 29 | bridge | hard | What Hindi thriller directed by Deepak SHivdasani features choreography by Phulw | Julie 2 | I don't have reliable information about this specific film. | unknown | 0.00 | 0.00 |
| 30 | bridge | hard | On the Knocking at the Gate in Macbeth is an essay by what English essayist best | Thomas Penson De Quincey | Thomas De Quincey | Thomas De Quincey  "On the Knocking at the Gate in Macbeth"  | 0.86 | 0.86 |
