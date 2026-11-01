-- One row per experiment arm: size, outcome rates, and pre-experiment balance checks.
SELECT
    segment                                   AS arm,
    count(*)                                  AS customers,
    avg(visit)                                AS visit_rate,
    avg(conversion)                           AS conversion_rate,
    avg(spend)                                AS spend_per_customer,
    sum(spend)                                AS total_spend,
    avg(spend) FILTER (WHERE conversion = 1)  AS spend_per_buyer,
    avg(history)                              AS avg_prior_year_spend,
    avg(recency)                              AS avg_months_since_purchase,
    avg(newbie)                               AS share_new_customers
FROM customers
GROUP BY segment
ORDER BY arm;
