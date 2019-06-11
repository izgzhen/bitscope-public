Related Work
==========

## Research Papers

+ Graph analysis
    - [O Bitcoin Where Art Thou? Insight into Large-Scale Transaction Graphs](https://pdfs.semanticscholar.org/96b9/5da0ab88de23641014abff2a5c0b5fec00c9.pdf) (zhen) (short, 2016)
        - In this demo, we present GraphSense, which is
            a solution that applies a graph-centric perspective on digital currency transactions. It allows users to explore transactions and follow the money flow, facilitates analytics by
            semantically enriching the transaction graph, supports path
            and graph pattern search, and guides analysts to anomalous
            data points. To deal with the growing volume and velocity of transaction data, we implemented our solution on a
            horizontally scalable data processing and analytics infrastructure.
        - [Ransomware Payments in the Bitcoin Ecosystem](https://arxiv.org/abs/1804.04080) (2018)
            - In this paper, we present a data-driven method for identifying and gathering information on Bitcoin transactions related to illicit activity based on footprints left on the public Bitcoin blockchain. We implement this method on-top-of the GraphSense open-source platform and apply it to empirically analyze transactions related to 35 ransomware families. We estimate the lower bound direct financial impact of each ransomware family and find that, from 2013 to mid-2017, the market for ransomware payments has a minimum worth of USD 12,768,536 (22,967.54 BTC). We also find that the market is highly skewed with only a few number of players responsible for the majority of the payments.
        - [Evolution of the Bitcoin Address Graph](https://aic.ai.wu.ac.at/~polleres/publications/filtz-etal-2017IDSC.pdf) (2017)
            - In this paper, we present initial results of a longitudinal
              study conducted over the Bitcoin address graph, which contains
              all addresses and transactions from the beginning of Bitcoin in
              January 2009 until 31st of August 2016. Our analysis reveals a
              highly-skewed degree distribution with a small number of
              outliers and illustrates that the entire graph is expanding rapidly.
              Furthermore, it demonstrates the power of address clustering
              heuristics for identifying real-world actors, who prefer to use
              Bitcoin for transferring rather than storing value.
    - [Bitcoin Transaction Graph Analysis](https://arxiv.org/abs/1502.01657) (by M Fleder - ‎2015 - ‎Cited by 71, short)
        - We explore the level of anonymity in the Bitcoin system. Our approach is two-fold: (i) We annotate the public transaction graph by linking bitcoin public keys to real people - either definitively or statistically. (ii) We run the annotated graph through our graph-analysis framework to find and summarize activity of both known and unknown users
    - [Exploring the Bitcoin Network](https://files.ifi.uzh.ch/stiller/CLOSER%202014/WEBIST/WEBIST/Society,%20e-Business%20and%20e-Government/Short%20Papers/WEBIST_2014_109_CR.pdf) (2014)
        - This explorative paper focuses on descriptive statistics and network analysis of the Bitcoin transaction graph
            based on recent data using graph mining algorithms. The analysis is carried out on different aggregations
            and subgraphs of the network. One important result concerns the relationship of network usage and
            exchange rate, where a strong connection could be confirmed. Moreover, there are indicators that the
            Bitcoin system is a “small world” network and follows a scale-free degree distribution. Furthermore, an
            example of how important network entities could be deanonymized is presented. Our study can serve as a
            starting point in investigating anonymity and economic relationships in Bitcoin on a new structural level. 
    - [Unsupervised Approaches to Detecting Anomalous Behavior in the Bitcoin Transaction Network](http://cs229.stanford.edu/proj2013/HirshmanHuangMacke-UnsupervisedApproachesToDetectingAnomalousBehaviorInTheBitcoinTransactionNetwork.pdf) (2013, short)
        - Unsupervised learning techniques revealed anomalies in a large bitcoin transaction network. We were able to identify certain users that conducted transactions in an atypical fashion, one that suggested some sort of money laundering.
        - Unfortunately, we have no way of proving our suspicions, as we do not have labeled data that points us to cases of these hypothesized mixing services. However, our work here could help pave the way for future clustering techniques, especially by allowing one to choose features that are more revealing of patterns in the data.
        - The unsupervised learning algorithms we applied, **K- means and RolX**, ended up achieving our intended ends of locating strange behavior in the network. Through clustering and role detection, we now have a much better idea of what to look for in a suspicious transaction, or, in particular, a string of suspicious transactions. Additional work should be done to both categorize and quantify these anomalies.
    - [Anomaly Detection in Bitcoin Network Using Unsupervised Learning Methods](https://arxiv.org/pdf/1611.03941.pdf) (short, 2017)
        - we use three unsupervised learning methods including k-means clustering, Mahalanobis distance, and Unsupervised Support Vector Machine (SVM) on two graphs generated by the Bitcoin transaction network: one graph has users as nodes, and the other has transactions as nodes.
    - [An Analysis of Anonymity in the Bitcoin System](https://arxiv.org/abs/1107.4524) (by F Reid - 2011 - Cited by 692)
        - In this chapter we consider the topological structure of two networks derived from Bitcoin's public transaction history. We show that the two networks have a non-trivial topological structure, provide complementary views of the Bitcoin system and have implications for anonymity. We combine these structures with external information and techniques such as **context discovery and flow analysis** to investigate an alleged theft of Bitcoins, which, at the time of the theft, had a market value of approximately half a million U.S. dollars.
    - [Quantitative Analysis of the Full Bitcoin Transaction Graph](https://eprint.iacr.org/2012/584.pdf) (by D Ron - 2012 - Cited by 564)
        - The Bitcoin scheme is a rare example of a large scale global payment system in which all the transactions are publicly accessible (but in an anonymous way). We downloaded the full history of this scheme, and analyzed many statistical properties of its associated transaction graph. In this paper we answer for the first time a variety of interest- ing questions about the typical behavior of users, how they acquire and how they spend their bitcoins, the balance of bitcoins they keep in their accounts, and how they move bitcoins between their various accounts in order to better protect their privacy. In addition, we isolated all the large transactions in the system, and discovered that almost all of them are closely related to a single large transaction that took place in November 2010, even though the associated users apparently tried to hide this fact with many strange looking long chains and fork-merge structures in the transaction graph.
    - **[Analyzing the Bitcoin Network: The First Four Years](https://www.mdpi.com/1999-5903/8/1/7/htm)** (zhitong) (by M Lischke - ‎2016 - ‎Cited by 63)
        - Abstract: In this explorative study, we examine the economy and transaction network of the decentralized digital currency Bitcoin during the first four years of its existence. The objective is to develop insights into the evolution of the Bitcoin economy during this period. For this, we establish and analyze a novel integrated dataset that enriches data from the Bitcoin blockchain with off-network data such as business categories and geo-locations. Our analyses reveal the major Bitcoin businesses and markets. Our results also give insights on the business distribution by countries and how businesses evolve over time. We also show that there is a gambling network that features many very small transactions. Furthermore, regional differences in the adoption and business distribution could be found. In the network analysis, the small world phenomenon is investigated and confirmed for several subgraphs of the Bitcoin network.
    - [Structure and Evolution of Bitcoin Transaction Network](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-7-final.pdf)
        - This work analyzes the transaction network to identify trends in the
            macroscopic evolution of the network structure over time. A time-history of descriptive network
            statistics was developed and used as input into a machine learning model of Bitcoin price. In order
            to explore the potential of using more microscopic network data, we proposed a deep learningbased representation embedding of the network structure that compressed the entire network into
            a 128-word vector. The performance of a basic neural network model trained using this embedding
            was compared to several baselines. We found that while the embedded representation does
            increase the performance of the network slightly, it is not the best representation for our current
            application.
    - [Cointopia: Blockchain Analysis using Online Forums](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-87-final.pdf)
        - In this paper we detail the results of exploratory data
            analysis against the Bitcoin blockchain, and compare various communities identified through mention and usage in
            online forums and datasets. By comparing these disparate
            communities, we manage to identify trends in convergence
            and adoption, cluster different addresses into groups with
            similar characteristics, and uniquely identify individuals
            that may be engaged in suspicious activity and merit further investigation.
    - [External influence on Bitcoin trust network structure](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-85-final.pdf)
        - Networks can express how much people trust or
            distrust each other. Trust between people may be influenced
            by factors such as relationship, reputation status, experience,
            stereotypes, media, information and governments between
            some. Bitcoin trust networks express how much people trust
            or distrust each other. Trust in Bitcoin trust network may be
            influenced by external factors like Bitcoin prices or breach of
            Bitcoin exchanges like Mt. Gox. One of the goals for this project
            is to find the effect of external influence to changes in the trust
            network. One hypothesis is that increase in positive external
            influence would increase the overall trust in the network, while
            a drop would decrease the trust. The change in prices may
            also change the network structure, for example, an hypothesis
            is that a drop in price would make the positive trust clusters
            turtle-up [1] to support each other. The second goal would be
            to create a model that can correctly predict the effects of a
            network property change in Bitcoin’s price.
    - [Network Analysis of Weighted Signed Bitcoin Networks](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-67-final.pdf)
        - In this paper, we will analyze the trust network
            formed by users of Bitcoin as weighted signed networks. Since anonymity is the main feature
            in bitcoin transactions, a user’s reputation needs to be qualified to avoid fraudulent transactions. Towards this goal, we will study signed, weighted bitcoin trust networks.
    - [Predicting Bitcoin Transactions with Network Analysis](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-65-final.pdf)
        - The aim of this paper is to predict the future number of transactions
            in the Bitcoin network based on the current network architecture. The task is similar to link
            prediction, but instead computes the expected number of edges in the network in a given time
            interval. We explored two major approaches: using handcrafted topological and non-topological
            features, and learning features in an unsupervised approach leveraging node2vec. We found
            that models learning over these features did not outperform a baseline regression based on node
            and edge counts, and discuss possible reasons why that was the case.
    - [Offline Detection of Influential Bitcoin Users](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-64-final.pdf)
        - Our aim for this project was to determine the most influential entities within the Bitcoin network using only the offline
            transactional data provided by the Hungarian Virtual Observatory (HVO). We started by finding several summary
            statistics, such as node / edge count, the size of each set according to the Bow-tie model, and average clustering
            coefficient. This gave us a good indication that the network was structured and not just a random graph. By exploiting
            the network structure, we were then able to figure out a set of users and addresses that belonged to miners. In addition,
            we plotted the degree proportionality for both the users and addresses network and discovered that they generally follow
            a power-law type distribution (interestingly enough, miners had higher degrees than their counterpart non-miners). Once
            we had a good sense for how the network was structured, we applied centrality measures, including eigenvector
            centrality, betweenness centrality, and the PageRank algorithm, in order to find the most influential nodes under each
            method. We encountered a few issues while attempting to calculate true betweenness centrality, primarily due to the size
            of each network and lack of compute resources. Unfortunately, an approximate algorithm yielded poor accuracy and we
            elected to discontinue pursuit of betweenness measures. With the remaining methods, we were nonetheless able to
            discover several addresses (the same address was often consistently ‘influential’ across different algorithms) that we
            were able to look up in [2]. This enabled us to confirm that each address was in fact “influential” - as they are owned by
            large exchanges or miners. The most influential turned out to be owned by the creator of Bitcoin, which is a very nice
            result. In conclusion, we achieved our goal to determine the most influential entities within the Bitcoin network using only
            offline transactional data and our results are consistent with the prior literature
    - [User Categorization and Community Detection in Bitcoin Network](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-62-final.pdf)
        - This project works on Bitcoin user categorization
            and community detection. A new user network generation
            method is presented to better contract user addresses. Users
            playing different roles are recognized with K-Means. Three
            different methods, K-Means, Node2Vector and Fiedler Vector
            Method are applied to analyze the network community structure. A major radiant community plus a minor community
            structure is detected.
    - [Trust in Bitcoin Exchange Networks](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-57-final.pdf)
        - We consider two marketplaces to exchange bitcoins and dollars, Bitcoin OTC and Bitcoin
            Alpha. In these settings, trust is a fundamental necessity since any user can initiate a transaction, receive money from the other user and never send money back in the other currency.
            To achieve a web of trust, users rate each other after successful or unsuccessful trades. We
            analyze these datasets using the fairness and goodness method from [5]. These two measures
            can be used to predict the weight of a given edge (the rating that u gives to v) with good
            accuracy. We show that whenever the edge is reciprocated, social interaction becomes a better
            predictor than goodness, and users apply the talion law: "an eye for an eye".
            We analyze the relationship between trust in these networks with the exchange price of bitcoin
            and find that there is no evident correlation between these two variables.
    - [Anomaly Detection in the Bitcoin System - A Network Perspective](http://snap.stanford.edu/class/cs224w-2014/projects2014/cs224w-20-final.pdf)
        - Our goal is to
            detect which users and transactions are the most suspicious;
            in this case, anomalous behavior is a proxy for suspicious
            behavior. To this end, we use the laws of power degree &
            densification and local outlier factor (LOF) method (which
            is proceeded by k-means clustering method) on two graphs
            generated by the Bitcoin transaction network: one graph
            has users as nodes, and the other has transactions as nodes.
            We remark that the methods used here can be applied to
            any type of setting with an inherent graph structure, including, but not limited to, computer networks, telecommunications networks, auction networks, security networks, social
            networks, Web networks, or any financial networks. We use
            the Bitcoin transaction network in this paper due to the
            availability, size, and attractiveness of the data set.
+ Address Classification / Clustering / Deanonymization
    - **[An Evaluation of Bitcoin Address Classification based on Transaction History Summarization](https://arxiv.org/abs/1903.07994)** (2019) (tianyi)
        - In this paper, we propose new features in addition to those commonly used in the literature to build a classification model for detecting abnormality of Bitcoin network addresses. These features include various high orders of moments of transaction time (represented by block height) which summarizes the transaction history in an efficient way. The extracted features are trained by supervised machine learning methods on a labeling category data set. The experimental evaluation shows that these features have improved the performance of Bitcoin address classification significantly. We evaluate the results under eight classifiers and achieve the highest Micro-F1/Macro-F1 of 87%/86% with **LightGBM**.
            - The entity-based classification, however, suffers from
            data imbalance and data scarcity. Therefore, we plan to do the
            experiment on a larger dataset [6] in the future work so as to
            evaluate the entity-based scheme.
    - [The Unreasonable Effectiveness of Address Clustering](https://ieeexplore.ieee.org/abstract/document/7816867) (2016)
        - Address clustering tries to construct the one-to-many mapping from entities to addresses in the Bitcoin system. Simple heuristics based on the micro-structure of transactions have proved very effective in practice. In this paper we describe the primary reasons behind this effectiveness: address reuse, avoidable merging, super-clusters with high centrality, the incremental growth of address clusters. We quantify their impact during Bitcoin's first seven years of existence.
            - Our future work revolves around the internal structure of address clusters, à la the bipartite graph in Fig. 6. This representation shows the structure of an address cluster beyond a simple set of addresses and may provide further insight into its formation and behaviour.
    - [BitIodine: Extracting Intelligence from the Bitcoin Network](https://link.springer.com/chapter/10.1007%2F978-3-662-45472-5_29) (by M Spagnuolo - 2014 - Cited by 144)
        - In this paper we present a modular framework, BitIodine, which parses the blockchain, clusters addresses that are likely to belong to a same user or group of users, classifies such users and labels them, and finally visualizes complex information extracted from the Bitcoin network. BitIodine labels users semi-automatically with information on their identity and actions which is automatically scraped from openly available information sources. BitIodine also supports manual investigation by finding paths and reverse paths between addresses or users. We tested BitIodine on several real-world use cases, identified an address likely to belong to the encrypted Silk Road cold wallet, or investigated the CryptoLocker ransomware and accurately quantified the number of ransoms paid, as well as information about the victims. We release a prototype of BitIodine as a library for building Bitcoin forensic analysis tools.
            - The main limitation is that the first heuristic presented in Sect. 3.1 works under the assumption that owners do not share private keys. This does not always hold: for example, some web wallets have pools that would be mistakenly grouped as a single user. This is why we defined the owns relation as   𝑜𝑤𝑛𝑠(𝑎𝑖)=𝑢𝑘  if and only if   𝑢𝑘  owns the private key of   𝑎𝑖 .
            - Moreover, the current implementation of the Classifier module needs to load the transaction graph and the clusters in memory, making classification a memory-intensive task. Also, BitIodine keeps data in two different fashions: in a relational database (the blockchain and features database) and in a graph (transaction and user graphs). This can be seen as redundant. In a future release, a single, efficient graph solution could replace the relational blockchain DB. In general, we see an on-disk graph database such as Neo4j needed if BitIodine is used in production, even with the drawbacks detailed in Sect. 3.2.
            - Furthermore, currently we label users in a (semi-)automated way by scraping information on known addresses from the web. In future extensions of this work, we envision to mine behavioral patterns of users on the network with unsupervised clustering or classification techniques.
    - [Breaking Bad: De-Anonymising Entity Types on the Bitcoin Blockchain Using Supervised Machine Learning](https://scholarspace.manoa.hawaii.edu/bitstream/10125/50331/1/paper0444.pdf)
        - This paper presents a novel approach for reducing the anonymity of the Bitcoin Blockchain by using Supervised Machine Learning to predict the type of yet-unidentified entities. We utilised a sample of 434 entities (with ≈ 200 million transactions), whose identity and type had been revealed, as training set data and built classifiers differentiating among 10 categories. Our main finding is that we can indeed predict the type of a yet-unidentified entity. Using the Gradient Boosting algorithm, we achieve an accuracy of 77% and F1-score of ≈ 0.75. We discuss our novel approach of Supervised Machine Learning for uncovering Bitcoin Blockchain anonymity and its potential applications to forensics and financial compliance and its societal implications, outline study limitations and propose future research directions.
            - In the future, we would seek to increase the relatively
            low sample size of identified clusters and add
            further cluster categories to create a more fine-grained
            differentiation between the clusters. Also, additional
            data could be utilised by harnessing more of the
            inherently available data on the Bitcoin Blockchain, as
            discussed in Sec. 4.2. Also, the feature engineering
            process could be improved, e.g. by using automated
            feature extraction. Lastly, we want to apply our
            model on the whole of Bitcoin Blockchain data and
            consequently present insights on the uncovered structure
            of the Bitcoin Blockchain, such as category distribution
            and transaction flow characteristics between those
            distributions.
    - [Deanonymization in the Bitcoin P2P Network](https://papers.nips.cc/paper/6735-deanonymization-in-the-bitcoin-p2p-network.pdf) (2017)
        + In this paper, we model the Bitcoin networking stack and
            analyze its anonymity properties, both pre- and post-2015. The core problem is
            one of epidemic source inference over graphs, where the observational model and
            spreading mechanisms are informed by Bitcoin’s implementation; notably, these
            models have not been studied in the epidemic source detection literature before.
            We identify and analyze near-optimal source estimators. This analysis suggests
            that Bitcoin’s networking protocols (both pre- and post-2015) offer poor anonymity
            properties on networks with a regular-tree topology. We confirm this claim in
            simulation on a 2015 snapshot of the real Bitcoin P2P network topology
    + [A New Approach to Deanonymization of Unreachable Bitcoin Nodes](https://eprint.iacr.org/2018/243.pdf) (2017)
        + In this paper, we devise an efficient method to link the sessions of unreachable nodes, even if they connect to the Bitcoin network over the
            Tor. We achieve this using a new approach based on organizing the blockrequests made by the nodes in a Bitcoin session graph. This attack also
            works against the modified Bitcoin client. We performed experiments on
            the Bitcoin main network, and were able to link consecutive sessions with
            a precision of 0.90 and a recall of 0.71. We also provide counter-measures
            to mitigate the attacks.
    + [Identification of High Yielding Investment Programs in Bitcoin via Transactions Pattern Analysis](https://ieeexplore.ieee.org/document/8254420/)
        + Although Bitcoin is one of the most successful decentralized cryptocurrency, recent research has revealed that it can be used as fraudulent activities such as HYIP (High Yield Investment Program). To identify such undesired activities, it is important to obtain Bitcoin addresses related with fraud. So far, the identification of such activities is based upon relating Bitcoin addresses with graph mining procedures. In this paper, we follow a different approach for identifying Bitcoin addresses related with HYIP by analyzing transactions patterns. In particular, based on the individual inspection of HYIP activity in Bitcoin, we propose a number of features that can be extracted from transactions. In particular, a signed integer called pattern is assigned to each transaction and the frequency of each pattern is calculated as key features. By evaluating the classification performance with more than 1,500 labeled Bitcoin addresses, it is shown that about 83% of HYIP addresses are correctly classified while maintaining false positive rate less than 4.4%.
    + [Automatic Bitcoin Address Clustering](https://bitfury.com/content/downloads/clustering_whitepaper.pdf)
        + In this
            paper, we propose to use off-chain information as votes for
            address separation and to consider it together with blockchain
            information during the clustering model construction step. Both
            blockchain and off-chain information are not reliable, and our
            approach aims to filter out errors in input data.
    + [Could Network Information Facilitate Address Clustering in Bitcoin?](https://fc17.ifca.ai/bitcoin/papers/bitcoin17-final11.pdf)
        + In this paper, we assess whether combining blockchain and network information may facilitate the clustering
            process. For this purpose, we apply all applicable clustering heuristics
            that are known to us to current blockchain information and associate
            the resulting clusters with IP address information extracted from observing the message flooding process of the bitcoin network. The results
            indicate that only a small share of clusters (less than 8 %) were conspicuously associated with a single IP address. Also, only a small number of
            IP addresses showed a conspicuous association with a single cluster.
    + [Data-Driven De-Anonymization in Bitcoin](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/155286/eth-48205-01.pdf)
        + We analyse the performance of several clustering algorithms in the digital peerto-peer currency Bitcoin. Clustering in Bitcoin refers to the task of finding
            addresses that belongs to the same wallet as a given address.
        + In order to assess the effectiveness of clustering strategies we exploit a vulnerability in the implementation of Connection Bloom Filtering to capture ground
            truth data about 37,585 Bitcoin wallets and the addresses they own. In addition
            to well-known clustering techniques, we introduce two new strategies, apply them
            on addresses of the collected wallets and evaluate precision and recall using the
            ground truth. Due to the nature of the Connection Bloom Filtering vulnerability
            the data we collect is not without errors. We present a method to correct the
            performance metrics in the presence of such inaccuracies.
        + Our results demonstrate that even modern wallet software can not protect its
            users properly. Even with the most basic clustering technique known as multiinput heuristic, an adversary can guess on average 68.59% addresses of a victim.
            We show that this metric can be further improved by combining several more
            sophisticated heuristics.
    + [User Categorization and Community Detection in Bitcoin Network](http://snap.stanford.edu/class/cs224w-2017/projects/cs224w-62-final.pdf)
        + This project works on Bitcoin user categorization and community detection.
        A new user network generation method is presented to better contract user addresses.
        Users playing different roles are recognized with K-Means.
        Three different methods, K-Means, Node2Vector and Fiedler Vector Method are applied to analyze the network community structure.
        A major radiant community plus a minor community structure is detected.
+ Economics

    - **[Cryptoeconomics: Data Application for TokenSales Analys](https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1001&context=icis2017b)** (2017)
        - We collected 500 ICO startups to build our database from the ICO-trackers. It consists of founder, white
            papers, likes and followers in Twitter, ICO results so on. We also collected USD exchange rates of
            cryptocurrencies to see the price effect of ICO investments. In this data, we considered top 10 ICO startups
            in 2017 as the successful cases. Using selected criteria, we modeled the prediction of success by the **logistic
            regression method**, which may solve our classification problem, successful or not.
    - <strikethrough>[The digital traces of bubbles: feedback cycles between socio-economic signals in the Bitcoin economy](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0623)</strikethrough> (zhen: this one looks a bit off-topic) (2014)
        - What is the role of social interactions in the creation of price bubbles? ... We thus quantify **four socio-economic signals** about Bitcoin from large datasets: price on online exchanges, volume of word-of-mouth communication in online social media, volume of information search and user base growth. By using **vector autoregression**, we identify two positive feedback loops that lead to price bubbles in the absence of exogenous stimuli: one driven by word of mouth, and the other by new Bitcoin adopters. We also observe that spikes in information search, presumably linked to external events, precede drastic price declines.
+ Trading Algo & Price Prediction
    - [Social signals and algorithmic trading of Bitcoin](https://royalsocietypublishing.org/doi/full/10.1098/rsos.150288) (2015)
        - In our analysis, we include **economic signals** of volume and price of exchange for USD, adoption of the Bitcoin technology and transaction volume of Bitcoin. We add **social signals** related to information search, word of mouth volume, emotional valence and opinion polarization as expressed in tweets related to Bitcoin for more than 3 years. Our analysis reveals that increases in opinion polarization and exchange volume precede rising Bitcoin prices, and that emotional valence precedes opinion polarization and rising exchange volumes. We apply these insights to design algorithmic trading strategies for Bitcoin, reaching very high profits in less than a year. We verify this high profitability with robust statistical methods that take into account risk and trading costs, confirming the long-standing hypothesis that trading-based social media sentiment has the potential to yield positive returns on investment.
    - [Statistical Analysis of the Exchange Rate of Bitcoin](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133678) (2015)
        - We provide a statistical analysis of the **log-returns** of the exchange rate of Bitcoin versus the United States Dollar. Fifteen of the most popular parametric distributions in finance are fitted to the log-returns. The generalized hyperbolic distribution is shown to give the best fit. Predictions are given for future values of the exchange rate.
    - [Bitcoin Price Prediction: An ARIMA Approach](https://arxiv.org/pdf/1904.05315.pdf) (2019)
        - Here, we aim at revealing the usefulness of
            traditional **autoregressive integrative moving average (ARIMA)
            model** in predicting the future value of bitcoin by analyzing
            the price time series in a 3-years-long time period. On the one
            hand, our empirical studies reveal that this simple scheme is
            efficient in sub-periods in which the behavior of the time-series
            is almost unchanged, especially when it is used for short-term
            prediction, e.g. 1-day. On the other hand, when we try to train
            the ARIMA model to a 3-years-long period, during which the
            bitcoin price has experienced different behaviors, or when we
            try to use it for a long-term prediction, we observe that it
            introduces large prediction errors. Especially, the ARIMA model
            is unable to capture the sharp fluctuations in the price, e.g. the
            volatility at the end of 2017. Then, it calls for more features to
            be extracted and used along with the price for a more accurate
            prediction of the price. We have further investigated the bitcoin
            price prediction using an ARIMA model, trained over a large
            dataset, and a limited test window of the bitcoin price, with
            length w, as inputs. Our study sheds lights on the interaction of
            the prediction accuracy, choice of (p, q, d), and window size w.
    - [Seq2Seq RNNs and ARIMA models for Cryptocurrency Prediction: A Comparative Study](https://fintech.kdd2018.a.intuit.com/papers/DSF2018_paper_papapetrou.pdf) (tianyi) (2018)
        - Cyrptocurrency price prediction has recently become an alluring
            topic, attracting massive media and investor interest. Traditional
            models, such as Autoregressive Integrated Moving Average models
            (ARIMA) and models with more modern popularity, such as **Recurrent Neural Networks** (RNN’s) can be considered candidates for
            such financial prediction problems, with RNN’s being capable of utilizing various endogenous and exogenous input sources. This study
            compares the model performance of ARIMA to that of a seq2seq recurrent deep multi-layer neural network (seq2seq) utilizing a varied
            selection of inputs types. The results demonstrate superior performance of seq2seq over ARIMA, for models generated throughout
            most of bitcoin price history, with additional data sources leading
            to better performance during less volatile price periods.
- Graph Representation Learning
    - https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf
    - https://github.com/williamleif/GraphSAGE
- Temporal Networks
    - [Motifs in Temporal Networks](https://cs.stanford.edu/people/jure/pubs/motifs-wsdm17.pdf)
        - Networks are a fundamental tool for modeling complex systems ina variety of domains including social and communication networksas well as biology and neuroscience.  Small subgraph patterns innetworks, called network motifs, are crucial to understanding thestructure  and  function  of  these  systems.    However,  the  role  ofnetwork motifs in temporal networks, which contain many times-tamped links between the nodes, is not yet well understood.
        - Here we develop a notion of a temporal network motif as an ele-mentary unit of temporal networks and provide a general method-ology for counting such motifs. We define temporal network motifsas induced subgraphs on sequences of temporal edges, design fastalgorithms for counting temporal motifs, and prove their runtimecomplexity. Our fast algorithms achieve up to 56.5x speedup com-pared to a baseline method. Furthermore, we use our algorithms tocount temporal motifs in a variety of networks.  Results show thatnetworks from different domains have significantly different mo-tif counts, whereas networks from the same domain tend to havesimilar motif counts.   We also find that different motifs occur atdifferent time scales, which provides further insights into structureand function of temporal networks.
        - https://snap.stanford.edu/temporal-motifs/data.html
    - [Temporal Graph Generation Based on a Distributionof Temporal Motifs](http://www.mlgworkshop.org/2018/papers/MLG2018_paper_42.pdf)
        - Generating a synthetic graph that is similar to a given real-world graph is a critical requirement for privacy preservationand benchmarking purposes. Various generative models at-tempt to generate static graphs similar to real-world graphs.However, generation of temporal graphs is still an open re-search area. We present a temporal-motif based approach togenerate synthetic temporal graph datasets and show resultsfrom three real-world use cases. We show that our approachcan generate high fidelity synthetic graph. We also showthat this approach can also generate multi-type heteroge-neous graph. We also present a parameterized version of ourapproach which can generate linear, sub-linear, and super-linear preferential attachment graph.
    - [Do the Rich Get Richer? An Empirical Analysis of the Bitcoin Transaction Network](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086197)
        - we reconstruct the network of transactions and extract the time and amount of each payment. We analyze the structure of the transaction network by measuring network characteristics over time, such as the degree distribution, degree correlations and clustering. We find that linear preferential attachment drives the growth of the network. We also study the dynamics taking place on the transaction network, i.e. the flow of money. We measure temporal patterns and the wealth accumulation. Investigating the microscopic statistics of money movement, we find that sublinear preferential attachment governs the evolution of the wealth distribution. We report a scaling law between the degree and wealth associated to individual nodes.
        - http://www.vo.elte.hu/bitcoin/default.htm
    - [Inferring the interplay between network structure and market effects in Bitcoin](https://iopscience.iop.org/1367-2630/16/12/125003/)
        - A main focus in economics research is understanding the time series of prices of goods and assets. While statistical models using only the properties of the time series itself have been successful in many aspects, we expect to gain a better understanding of the phenomena involved if we can model the underlying system of interacting agents. In this article, we consider the history of Bitcoin, a novel digital currency system, for which the complete list of transactions is available for analysis. Using this dataset, we reconstruct the transaction network between users and analyze changes in the structure of the subgraph induced by the most active users. Our approach is based on the unsupervised identification of important features of the time variation of the network. Applying the widely used method of Principal Component Analysis to the matrix constructed from snapshots of the network at different times, we are able to show how structural changes in the network accompany significant changes in the exchange price of bitcoins.
    - [The Rise and Fall of Network Stars: Analyzing 2.5 Million Graphs to Reveal How High-Degree Vertices Emerge over Time](https://arxiv.org/pdf/1706.06690.pdf)
        - Trends change rapidly in today’s world,  prompting this key question:  What is the mechanism behind theemergence of new trends? By representing real-world dynamic systems as complex networks, the emergenceof new trends can be symbolized by vertices that “shine.” That is, at a specific time interval in a network’s life,certain vertices become increasingly connected to other vertices. This process creates new high-degree vertices,i.e., network stars. Thus, to study trends, we must look at how networks evolve over time and determine howthe stars behave.  In our research, we constructed the largest publicly available network evolution dataset todate, which contains 38,000 real-world networks and 2.5 million graphs. Then, we performed the first precisewide-scale analysis of the evolution of networks with various scales. Three primary observations resulted: (a)links are most prevalent among vertices that join a network at a similar time; (b) the rate that new verticesjoin a network is a central factor in molding a network’s topology; and (c) the emergence of network stars(high-degree vertices) is correlated with fast-growing networks. We applied our learnings to develop a flexiblenetwork-generation model based on large-scale, real-world data. This model gives a better understanding ofhow stars rise and fall within networks, and is applicable to dynamic systems both in nature and society
    - [Predicting User Performance and Bitcoin Price Using Block Chain Transaction Network](https://arxiv.org/pdf/1804.08044.pdf)
        - This work addresses several questions about the Bitcoin network.We have shown that, the users in the Bitcoin network tend to reuse their addresses thatwould expose theusers. We showed that the percentage of addresses reused rtimes in a transaction is given by p = a r -2.5. The users of the Bitcoin network are classified into sellers and customers based on a Z-score obtained from the study of a random graph. This kind of technique could be used to classify nodes and observe different group behaviors. For example in this work we showed that the ratio between customer population and seller population is very important factor and has a high correlation with bitcoin price. We have also shown that for the sellersthe amount of money they earn is correlated with their page rank in the graph. For future work one could study other centrality measures to see how it is correlated with different properties of nodes. Also one could use these parameters as a feature vector to predict bitcoin value.
+ Market Analysis & Fraud Detection
    + [Price manipulation in the Bitcoin ecosystem](https://www.sciencedirect.com/science/article/pii/S0304393217301666) (by N Gandal - ‎2018 - ‎Cited by 101)
        + This paper identifies and analyzes the impact of suspicious trading activity on the Mt. Gox Bitcoin currency exchange, in which approximately 600,000 bitcoins (BTC) valued at $188 million were fraudulently acquired. During both periods, the USD-BTC exchange rate rose by an average of four percent on days when suspicious trades took place, compared to a slight decline on days without suspicious activity. Based on rigorous analysis with extensive robustness checks, the paper demonstrates that the suspicious trading activity likely caused the unprecedented spike in the USD-BTC exchange rate in late 2013, when the rate jumped from around $150 to more than $1,000 in two months.
    + [Fraud Detection in Signed Bitcoin Trading Platform Networks](http://web.stanford.edu/class/cs224w/reports/CS224W-2018-101.pdf) (2018)
        + https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
    + **[Data mining for detecting Bitcoin Ponzi schemes](https://arxiv.org/pdf/1803.00646.pdf)** (2018)
        + We apply data mining techniques to
            detect Bitcoin addresses related to Ponzi schemes. Our starting
            point is a dataset of features of real-world Ponzi schemes,
            that we construct by analysing, on the Bitcoin blockchain, the
            transactions used to perform the scams. We use this dataset to
            experiment with various machine learning algorithms, and we
            assess their effectiveness through standard validation protocols
            and performance metrics. The best of the classifiers we have
            experimented can identify most of the Ponzi schemes in the
            dataset, with a low number of false positives
    + [Flash Boys 2.0: Frontrunning, Transaction Reordering, and Consensus Instability in Decentralized Exchanges](https://arxiv.org/pdf/1904.05234.pdf) (2019)
        + Blockchains, and specifically smart contracts, have
            promised to create fair and transparent trading ecosystems.
            Unfortunately, we show that this promise has not been met. We
            document and quantify the widespread and rising deployment of
            arbitrage bots in blockchain systems, specifically in decentralized
            exchanges (or “DEXes”). Like high-frequency traders on Wall
            Street, these bots exploit inefficiencies in DEXes, paying high
            transaction fees and optimizing network latency to frontrun, i.e.,
            anticipate and exploit, ordinary users’ DEX trades.
        +  We study the breadth of DEX arbitrage bots in a subset of
            transactions that yield quantifiable revenue to these bots. We also
            study bots’ profit-making strategies, with a focus on blockchainspecific elements. We observe bots engage in what we call priority
            gas auctions (PGAs), competitively bidding up transaction fees in
            order to obtain priority ordering, i.e., early block position and
            execution, for their transactions. PGAs present an interesting
            and complex new continuous-time, partial-information, gametheoretic model that we formalize and study. We release an
            interactive web portal, frontrun.me, to provide the community
            with real-time data on PGAs.
        + We additionally show that high fees paid for priority transaction ordering poses a systemic risk to consensus-layer security.
        + We explain that such fees are just one form of a general phenomenon in DEXes and beyond—what we call miner extractable
            value (MEV)—that poses concrete, measurable, consensus-layer
            security risks. We show empirically that MEV poses a realistic
            threat to Ethereum today.
        + Our work highlights the large, complex risks created by
            transaction-ordering dependencies in smart contracts and the
            ways in which traditional forms of financial-market exploitation
            are adapting to and penetrating blockchain economies.
+ Survey
    + [Big-Crypto: Big Data, Blockchain and Cryptocurrency](https://www.mdpi.com/2504-2289/2/4/34) (2018)

## Posts/Books

- https://medium.com/@fisherinfo/fraud-detection-and-cryptocurrency-489c3663bfd0
    - So we applied our algorithms to 150 ICOs:
      39 of 150 ICO wallets detected anomalies ~ 26 %
      ~20000 transactions are classified as suspicious
      Total Amount of ETH in suspicious transactions = 60 000 ETHs
      1 of 4 ICOs possibly tried to create fake demand on their tokens during crowdsale process.
- https://medium.com/@sylvainartplayribes/chasing-fake-volume-a-crypto-plague-ea1a3c1e0b5e
    - In this piece I will expose why I believe more than $3 billion of all cryptoassets’ volume to be fabricated, and how OKex, #1 exchange rated by volume, is the main offender with up to 93% of its volume being nonexistent. I’ll endeavour to prove it by analyzing publicly available data
- https://towardsdatascience.com/detecting-financial-fraud-using-machine-learning-three-ways-of-winning-the-war-against-imbalanced-a03f8815cce9
    - Key Takeaways
        + Imbalanced data can be a serious problem for building predictive models, as it can affect our prediction capabilities and mask the fact that our model is not doing so good
        + Imblearn provides some great functionality for dealing with imbalanced data
        + Depending on your data, SMOTE, RandomUnderSampler or SMOTE + ENN techniques could be used. Each approach is different and it is really the matter of understanding which of them makes more sense for your situation.
        + It is important considering the trade-off between precision and recall and deciding accordingly which of them to prioritize when possible, considering possible business outcomes.
- https://medium.com/@huobiresearch/huobi-blockchain-data-analytics-vol-2-data-driven-analysis-of-large-bitcoin-transactions-f8f9ce698d8e
    - By building BTC nodes and capturing the top 1000 bitcoin transactions records from Jan 1, 2017 to date, this article builds upon transaction analysis to provide observation summaries and insights.
- [Handbook of Digital Currency: Chap 5](https://books.google.com/books?hl=en&lr=&id=RfWcBAAAQBAJ&oi=fnd&pg=PP1&dq=cryptocurrency+data+analysis&ots=2MrPLdz7wB&sig=7eDD2OBdLbCSJVPewQ19PpNrGWk#v=onepage&q=cryptocurrency%20data%20analysis&f=false)

## Reports/News

- https://www.wsj.com/articles/most-bitcoin-trading-faked-by-unregulated-exchanges-study-finds-11553259600


## Tools/Platforms

- http://graphsense.info
- https://www.blockchaintransparency.org

## Notebooks/Repos

- https://www.kaggle.com/wprice/bitcoin-mining-pool-classifier
    + This notebook uses transaction activity features to predict whether
      or not an address belongs to a miner
- https://github.com/Iwontbecreative/Bitcoin-adress-clustering
    + They validate raw transaction history for consistency, and group
      public keys (or address, _in some sense_) into cluster based on
      a set of deterministic rules:
    + rule 1: for any tx with one input and one output, the PKIDs of I and O
      are considered belonging to the same cluster
    + rule 2: for any tx with more than one inputs, all inputs are considered
      belonging to the same cluster
    + result: 174120 PKIDs is grouped into 109721 clusters
- https://github.com/rajathalex/BitcoinClustering
    + Uses the same dataset as the previous one
- https://github.com/pankh13/Bitcoin-Miner-Behavioral-Research
- https://github.com/archienorman11/thesis-bitcoin-clustering
    - This implementation seems including code for dissecting bitcoin blockchain raw data
    - https://github.com/archienorman11/thesis-bitcoin-clustering/blob/master/report.pdf
- https://github.com/mikispag/bitiodine
    - The [Features database](https://static.miki.it/pdf/thesis.pdf) seems very useful
