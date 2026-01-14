#项目介绍
    本项目基于likert量表评分机制，用于针对不同llm针对需求的评分数值进行量化评估
#版本内容更新
###### v1.0 ：
    1.kripp_alpha_ordinal（Krippendorff's α ，序数距离Ordinal）
    2.friedman_chi2（Friedman 非参数检验统计量）
    3.kendall_W（Ke4.ndall’s W，肯德尔和谐系数）
    4.MAE_to_median（距离中位数的平均绝对偏差）
    证明不同llm针对案例二需求的评分标准存在显著性差异
###### v1.1 ：
    新增：
    1.评分主要趋势
        1.1 trend_dim_req：按维度汇总共识中位数（平均/中位）、低分比例、平均分歧（range/IQR）
        1.2 trend_item_req：按条目汇总平均共识分、最弱维度、最大分歧维度
        1.3 consensus_req：落到每个 (item, dimension) 的共识/分歧细节（用于“主要问题集中在哪些条目/维度”）
    2.差异具体是什么
        2.1 bias_req_by_dim：各模型在各维度上相对共识的系统偏置（严/宽）
        2.2 rankcorr_req：各维度上模型两两在 item 排序上的相关（Spearman/Kendall），用来区分“只是偏严/偏松”还是“连相对判断都不同”
        2.3 top_dev_req：列出偏离共识最大的格子（差异发生在哪里、差了几分）
    3.离群模型（谁和其他模型差异最大）
        3.1 distmat_req / distpair_req / avgdist_req：两两 MAE 距离矩阵 + 每个模型到其他模型的平均距离（距离最大者更离群）
        3.2 loo_req：leave-one-out 的分歧贡献度（去掉某模型总体分歧下降最多者，是“拉大分歧”的主要贡献者）
    保留已有的(矩阵、共识、a、Friedman、 rater_sumary)基础上，新增"共同趋势/差异结构/离群模型"所需的可解释指标与Exce sheet 输出