-- Visit rate by arm within each customer segment, and the lift of each email vs. no email.
WITH segments AS (
    SELECT 'Prior-year spend' AS dimension, history_segment AS level, segment, visit, spend FROM customers
    UNION ALL SELECT 'Channel', channel, segment, visit, spend FROM customers
    UNION ALL SELECT 'Area', zip_code, segment, visit, spend FROM customers
    UNION ALL SELECT 'Customer type', CASE WHEN newbie = 1 THEN 'New (last 12 months)' ELSE 'Existing' END,
                     segment, visit, spend FROM customers
    UNION ALL SELECT 'Bought before', CASE WHEN mens = 1 AND womens = 1 THEN 'Both'
                                           WHEN mens = 1 THEN 'Men''s only' ELSE 'Women''s only' END,
                     segment, visit, spend FROM customers
),
rates AS (
    SELECT dimension, level, segment, count(*) AS n, avg(visit) AS visit_rate, avg(spend) AS spend
    FROM segments
    GROUP BY ALL
)
SELECT
    t.dimension,
    t.level,
    t.segment                         AS arm,
    t.n                               AS customers_in_arm,
    c.n                               AS customers_in_control,
    t.visit_rate - c.visit_rate       AS visit_lift,
    t.spend - c.spend                 AS spend_lift
FROM rates t
JOIN rates c ON c.dimension = t.dimension AND c.level = t.level AND c.segment = 'No E-Mail'
WHERE t.segment <> 'No E-Mail'
ORDER BY t.dimension, t.level, arm;
