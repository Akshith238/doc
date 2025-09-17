-- Q1: Hospital Patient Records Analysis
type Patient = (String, Int, Int)

samplePatients :: [Patient]
samplePatients = 
  [("Alice",25,1),("Bob",42,2),("Carol",35,3),("David",18,1),("Eve",50,2)]

countReasons :: [Patient] -> (Int, Int, Int, Int)
countReasons [] = (0,0,0,0)
countReasons ((_,age,reason):xs) =
  let (c1,c2,c3,ad) = countReasons xs
      adultInc = if age >= 18 then 1 else 0
  in case reason of
       1 -> (c1+1,c2,c3,ad+adultInc)
       2 -> (c1,c2+1,c3,ad+adultInc)
       3 -> (c1,c2,c3+1,ad+adultInc)
       _ -> (c1,c2,c3,ad+adultInc)

main :: IO ()
main = do
  let (g,e,s,ad) = countReasons samplePatients
  putStrLn "Hospital Patient Records Analysis"
  putStrLn $ "General Checkup: " ++ show g
  putStrLn $ "Emergency: " ++ show e
  putStrLn $ "Surgery: " ++ show s
  putStrLn $ "Total Adults: " ++ show ad

-- Q2: Cinema Ticket Sales Report
type Sale = (String, Int)

sampleSales :: [Sale]
sampleSales = [("Adult",5),("Child",3),("Senior",2),("Adult",4)]

sumSales :: [Sale] -> (Int,Int,Int,Int)
sumSales [] = (0,0,0,0)
sumSales ((cat,qty):xs) =
  let (a,c,s,revenue) = sumSales xs
  in case cat of
       "Adult"  -> (a+qty,c,s,revenue + qty*12)
       "Child"  -> (a,c+qty,s,revenue + qty*8)
       "Senior" -> (a,c,s+qty,revenue + qty*10)
       _        -> (a,c,s,revenue)

main :: IO ()
main = do
  let (a,c,s,revenue) = sumSales sampleSales
  putStrLn "Cinema Ticket Sales Report"
  putStrLn $ "Adult tickets: " ++ show a
  putStrLn $ "Child tickets: " ++ show c
  putStrLn $ "Senior tickets: " ++ show s
  putStrLn $ "Total revenue: $" ++ show revenue

-- Q3: Student Academic Performance Report
type Student = (String, Int)

sampleStudents :: [Student]
sampleStudents = 
  [("John",35),("Mary",78),("Paul",60),("Lucy",85),("Sam",55)]

classify :: Int -> String
classify m
  | m < 40 = "Fail"
  | m < 60 = "Pass"
  | m < 80 = "Merit"
  | otherwise = "Distinction"

categorize :: [Student] -> [(String,Int,String)]
categorize [] = []
categorize ((name,mark):xs) = 
  (name, mark, classify mark) : categorize xs

countPasses :: [Student] -> Int
countPasses [] = 0
countPasses ((_,mark):xs)
  | mark >= 40 = 1 + countPasses xs
  | otherwise  = countPasses xs

main :: IO ()
main = do
  let categorized = categorize sampleStudents
      passCount = countPasses sampleStudents
  putStrLn "Student Academic Performance Report"
  mapM_ print categorized
  putStrLn $ "Total Passed: " ++ show passCount
