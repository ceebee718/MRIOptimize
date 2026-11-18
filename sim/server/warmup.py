"""Write a small sequence + phantom so the Julia worker can compile before the first request."""
import sys
from .seqgen import build
from .app import write_phantom

def make(tag):
    p = dict(seq="TSE", b0=1.5, TR=1000, TE=40, flip=90, refoc=150, ETL=8, fovR=380, fovP=380,
             Nr=64, Np=64, thk=6, NEX=1, BW=300, R=1)
    seq, _, _ = build(p)
    seq.write(tag + ".seq")
    write_phantom(tag + ".phantom", "abdomen", dict(p, Nr=48, Np=48), False)

if __name__ == "__main__":
    make(sys.argv[1])
