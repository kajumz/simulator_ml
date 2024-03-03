WITH sub AS (
    SELECT
        product_name,
        toMonday(dt) AS monday,
        max(price) AS max_price,
        count(*) AS y,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 1 PRECEDING AND 1 preceding) AS y_lag_1,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING) AS y_lag_2,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 3 PRECEDING AND 3 PRECEDING) AS y_lag_3,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 4 PRECEDING AND 4 PRECEDING) AS y_lag_4,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 5 PRECEDING AND 5 PRECEDING) AS y_lag_5,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 6 PRECEDING AND 6 PRECEDING) AS y_lag_6
    FROM
        default.data_sales_train
    GROUP BY
        product_name,
        monday
),
aggregated AS (
    SELECT
        monday,
        SUM(y_lag_1) AS y_all_lag_1,
        SUM(y_lag_2) AS y_all_lag_2,
        SUM(y_lag_3) AS y_all_lag_3,
        SUM(y_lag_4) AS y_all_lag_4,
        SUM(y_lag_5) AS y_all_lag_5,
        SUM(y_lag_6) AS y_all_lag_6
    FROM 
        sub
    GROUP BY
        monday
)
SELECT
    sub.product_name,
    sub.monday,
    sub.max_price,
    sub.y,
    sub.y_lag_1,
    sub.y_lag_2,
    sub.y_lag_3,
    sub.y_lag_4,
    sub.y_lag_5,
    sub.y_lag_6,
    AVG(sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3) OVER (PARTITION BY sub.product_name ORDER BY sub.monday ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING) AS y_avg_3,
    MAX(sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3) OVER (PARTITION BY sub.product_name ORDER BY sub.monday ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING) AS y_max_3,
    MIN(sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3) OVER (PARTITION BY sub.product_name ORDER BY sub.monday ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING) AS y_min_3,
    AVG(sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3 + sub.y_lag_4 + sub.y_lag_5 + sub.y_lag_6) OVER (PARTITION BY sub.product_name ORDER BY sub.monday ROWS BETWEEN 5 PRECEDING AND 5 PRECEDING) AS y_avg_6,
    MAX(sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3 + sub.y_lag_4 + sub.y_lag_5 + sub.y_lag_6) OVER (PARTITION BY sub.product_name ORDER BY sub.monday ROWS BETWEEN 5 PRECEDING AND 5 PRECEDING) AS y_max_6,
    MIN(sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3 + sub.y_lag_4 + sub.y_lag_5 + sub.y_lag_6) OVER (PARTITION BY sub.product_name ORDER BY sub.monday ROWS BETWEEN 5 PRECEDING AND 5 PRECEDING) AS y_min_6,
    aggregated.y_all_lag_1,
    aggregated.y_all_lag_2,
    aggregated.y_all_lag_3,
    aggregated.y_all_lag_4,
    aggregated.y_all_lag_5,
    aggregated.y_all_lag_6
FROM sub 
JOIN aggregated ON sub.monday = aggregated.monday;
