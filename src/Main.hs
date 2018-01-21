module Main where

import Net
import Data.Traversable
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector.Storable as V
import System.Random
import Data.List
import Control.Monad

testNet :: DoubleNet -> [Vector Double] -> [Vector Double] -> Double
testNet n ins ans =
  let outs :: [Vector Double]
      outs = map (runNet n) ins
  in sum (zipWith (\x y -> norm_1 $ add x (scale (-1) y)) outs ans)

rankNets :: [DoubleNet] -> IO [(Double,DoubleNet)]
rankNets nets =
  do values <- mapM (\_ -> do x <- randomRIO (0,10)
                              return $ vector [x]) [1..100]
     let answers = map sin values
         scoredNets = map (\x -> (testNet x values answers,x)) nets
     return $ sortOn fst scoredNets

plotSome :: DoubleNet -> IO ()
plotSome n =
  do let pts = map (\x -> (x/1000)*10) $ [1..1000]
     mapM_ (\x -> putStrLn $ (show $ runNet n (vector [x])) ++ " " ++ (show $ (sin x)/2)) pts

sinFunc = DiffFunction sin (\x y -> y*cos x)

errFunc :: Double -> DifferentiableFunction (Vector Double) Double
errFunc c = DiffFunction (\v -> ((atIndex v 0) - c)^2) (\v w -> 2*((atIndex v 0) - c)*(atIndex w 0))

iterateNet :: Int -> DoubleNet -> IO DoubleNet
iterateNet 0 net = return net
iterateNet steps net =
  do let n = 10
     xs <- replicateM n $ randomRIO (0,10)

     --let x = 0.5
     grads <- forM xs (\x -> do let (val,err,grad) = runAndDiffWithError net (vector [x]) (errFunc $ (sin x)/2)
                                print (x,val,sin x,err)
                                return grad)
     let grad = foldl1 (zipWith (\a b -> (add (scale (1/fromIntegral n) b) a))) grads
     let newNet = updateNet net grad (-0.005)
     iterateNet (steps-1) newNet

main :: IO ()
main = do randNet <- randomSimpleNet 1 1 100
          finalNet <- iterateNet 100000 randNet
          plotSome finalNet
