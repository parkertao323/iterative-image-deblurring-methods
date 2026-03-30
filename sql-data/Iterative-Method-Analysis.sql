/* The Python adaptation is for convenient data presentation.I used sqliteviz as the primary SQL editor.

/*
0.Transform the CSV file to one that can be processed by a SQL editor. 


DROP VIEW IF EXISTS clean_metrics;

CREATE VIEW clean_metrics AS
WITH base AS (
    SELECT
        rowid AS rid,
        NULLIF(TRIM(Algorithm), '') AS Algorithm,
        CAST(Iteration AS INTEGER) AS Iteration,
        CAST(Residual AS REAL) AS Residual,
        CAST(PSNR AS REAL) AS PSNR,
        CAST(SSIM AS REAL) AS SSIM
    FROM Iterative_Algorithm_outputs
    WHERE Iteration IS NOT NULL
),
filled AS (
    SELECT
        b1.rid,
        (
            SELECT b2.Algorithm
            FROM base b2
            WHERE b2.rid <= b1.rid
              AND b2.Algorithm IS NOT NULL
            ORDER BY b2.rid DESC
            LIMIT 1
        ) AS Algorithm,
        b1.Iteration,
        b1.Residual,
        b1.PSNR,
        b1.SSIM
    FROM base b1
)
SELECT
    Algorithm,
    Iteration,
    Residual,
    PSNR,
    SSIM
FROM filled
WHERE Algorithm IS NOT NULL;

SELECT Algorithm, COUNT(*) AS n_rows
FROM clean_metrics
GROUP BY Algorithm;


/*
1.Find the maximum Pixel Signal-to-Noise Ratio(PSNR) of each algorithm
*/
SELECT Algorithm, Iteration, PSNR
FROM (
    SELECT
        Algorithm,
        Iteration,
        PSNR,
        ROW_NUMBER() OVER (PARTITION BY Algorithm ORDER BY PSNR DESC) AS row_num
    FROM clean_metrics
)
WHERE row_num = 1;


/*
2.Find the maximum Structural Similarity Index Measure(SSIM) of each algorithm
*/
SELECT Algorithm, Iteration, SSIM
FROM (
    SELECT
        Algorithm,
        Iteration,
        SSIM,
        ROW_NUMBER() OVER (PARTITION BY Algorithm ORDER BY SSIM DESC) AS row_num
    FROM clean_metrics
)
WHERE row_num = 1;


/*
3.First iteration where PSNR reaches a set value, 25 in this case
*/
SELECT Algorithm, MIN(Iteration) AS First_Iter_PSNR_Greater_25
FROM clean_metrics
WHERE PSNR >= 25
GROUP BY Algorithm;


/*
4.First iteration where residual becomes small enough,take 2 in this case
*/
SELECT Algorithm, MIN(Iteration) AS First_Iter_Residual_Small_2
FROM clean_metrics
WHERE Residual <= 2
GROUP BY Algorithm;


/*
5.Compare the best PSNR with the last PSNR in each algorithm
*/
WITH ranked AS (
    SELECT
        Algorithm,
        Iteration,
        PSNR,
        MAX(Iteration) OVER (PARTITION BY Algorithm) AS Last_Iter
    FROM clean_metrics
)
SELECT
    Algorithm,
    MAX(PSNR) AS Best_PSNR,
    MAX(CASE WHEN Iteration = Last_Iter THEN PSNR END) AS LAST_PSNR
FROM ranked
GROUP BY Algorithm;


/*
6.Compare the best SSIM with the last SSIM in each algorithm
*/
WITH ranked AS (
    SELECT
        Algorithm,
        Iteration,
        SSIM,
        MAX(Iteration) OVER (PARTITION BY Algorithm) AS Last_Iter
    FROM clean_metrics
)
SELECT
    Algorithm,
    MAX(SSIM) AS Best_SSIM,
    MAX(CASE WHEN Iteration = Last_Iter THEN SSIM END) AS LAST_SSIM
FROM ranked
GROUP BY Algorithm;


/*
7.Locate the iteration where each algorithm meets its best PSNR, and present the corresponding residual and SSIM
*/
SELECT Algorithm, Iteration, PSNR, Residual, SSIM
FROM (
    SELECT
        Algorithm,
        Iteration,
        PSNR,
        Residual,
        SSIM,
        ROW_NUMBER() OVER (PARTITION BY Algorithm ORDER BY PSNR DESC) AS row_num
    FROM clean_metrics
)
WHERE row_num = 1;


/*
8.Similarly, locate the iteration where each algorithm meets its best SSIM, and present the corresponding residual and PSNR
*/
SELECT Algorithm, Iteration, SSIM, Residual, PSNR
FROM (
    SELECT
        Algorithm,
        Iteration,
        SSIM,
        Residual,
        PSNR,
        ROW_NUMBER() OVER (PARTITION BY Algorithm ORDER BY SSIM DESC) AS row_num
    FROM clean_metrics
)
WHERE row_num = 1;
