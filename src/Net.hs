{-# LANGUAGE DeriveFunctor #-}
module Net where

import Prelude hiding (id,(.))

import Numeric.LinearAlgebra hiding (range)
import Data.Traversable
import System.Random
import qualified Data.Vector.Storable as V
import Data.Array

import Debug.Trace
import Control.Category

--A function with a deriviative
--a and b are assumed to be numbers, and thus their own tangent/cotangent spaces
data DifferentiableFunction a b =
  DiffFunction { dfRun :: (a -> b)
               , dfDerivative :: (a -> (a -> b))
               }


type DF = DifferentiableFunction

instance Category DifferentiableFunction where
  id = DiffFunction id (const id)
  (DiffFunction f df) . (DiffFunction g dg) =
    DiffFunction (f . g) (\x -> (df (g x)) . (dg x))

simpleDiff :: (Num a) => DifferentiableFunction a b -> (a -> b)
simpleDiff df = (\x -> dfDerivative df x 1)

{-
instance Category DifferentiableFunction where
  id = DiffFunction id id
  (DiffFunction f df) . (DiffFunction g dg) = DiffFunction (f . g) ((f . dg)??df)
-}
data Layer c = Layer
  { layerWeights :: c
  , layerActivation :: DifferentiableFunction Double Double
  } deriving (Functor)

instance (Show c) => Show (Layer c) where
  show = show . layerWeights

data Net c = Net
  { netLayerSizes :: [Int]
  , netContent :: [Layer c]} deriving (Functor,Show)

netParameters :: Net c -> [c]
netParameters = (map layerWeights) . netContent

type NetShape = Net ()
type DoubleNet = Net (Matrix Double)


runLayer :: Layer (Matrix Double) -> Vector Double -> Vector Double
runLayer Layer{layerWeights=inputs,layerActivation=act} v =
  V.map (dfRun act) $ inputs #> v

runNet :: Net (Matrix Double) -> Vector Double -> Vector Double
runNet net input =
  --first layer needs a fake input that's always 1
  let realInput = vjoin [input,1.0]
  in foldl (flip runLayer) realInput (netContent net)


runAndDiffLayer :: Layer (Matrix Double) -> Vector Double -> (Vector Double,Array (Int,Int) (Vector Double),Matrix Double)
runAndDiffLayer Layer{layerWeights=weights,layerActivation=act} input =
  let dact = simpleDiff act
      linearValues = weights #> input
      output = V.map (dfRun act) linearValues
      (nR,nC) = size weights
      limit = (nR-1,nC-1)
      createSingle n i x = vector $ replicate i 0 ++ [x] ++ replicate (n-i-1) 0
      derivArray =
        listArray ((0,0),limit) $
        (\(r,c) -> createSingle nR r (atIndex input c)) <$> range ((0,0),limit)
  in (output,derivArray,(diag (V.map dact linearValues)) <> weights)

runAndDiffNet :: Net (Matrix Double) -> Vector Double -> (Vector Double,[Array (Int,Int) (Vector Double)])
runAndDiffNet net input =
  let realInput = vjoin [input,1.0]
  in foldl (\(newIn,prevDiff) l ->
               let (nextIn,layerDiff,layerVDiff) = (runAndDiffLayer l newIn)
               in (nextIn,((fmap (layerVDiff #>)) <$> prevDiff) ++ [layerDiff]))
     (realInput,[]) $ netContent net

runAndDiffWithError :: Net (Matrix Double) -> Vector Double -> DifferentiableFunction (Vector Double) Double -> (Vector Double,Double,[Matrix Double])
runAndDiffWithError net input errFunc =
  let (out,grad) = runAndDiffNet net input
      derr = dfDerivative errFunc
  in (out,dfRun errFunc out,((fromArray2D . fmap (\v -> (derr out v))) <$> grad))

modifyNet :: Net a -> [b] -> Net b
modifyNet net contents =
  net{netContent = zipWith (\m l -> m <$ l) contents $ netContent net}

updateNet :: Net (Matrix Double) -> [Matrix Double] -> Double -> Net (Matrix Double)
updateNet net grad c =
  let param = netParameters net
      newParams = zipWith (\oldM gradM -> oldM + scale c gradM) param grad
  in modifyNet net newParams

tanhFunc :: DifferentiableFunction Double Double
tanhFunc = DiffFunction f (\y c -> c*(let z = f y in 1-z^2))
  where
    f x = (2/(1+(exp (-2*x))) - 1)

--layerAsDiffFunc :: Layer (Matrix Double) -> DifferentiableFunction (Matrix Double,Vector Double) (Vector Double)

--netAsDiffFunction :: Net (Matrix Double) -> DifferentiableFunction ([Matrix Double],Vector Double) (Vector Double)

randomSimpleNet :: Int -> Int -> Int -> IO (Net (Matrix Double))
randomSimpleNet numIn numOut internal=
  do layer1 <- Layer <$> (scale (1/(fromIntegral (numIn+1))) <$> randn internal (numIn+1)) <*> pure tanhFunc

     layer2 <- Layer <$> (scale (1/(fromIntegral internal)) <$> randn numOut internal) <*> pure tanhFunc

     return $ Net [numIn,internal,numOut] [layer1,layer2]
