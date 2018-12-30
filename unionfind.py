"""
MakeSet(x) initializes disjoint set for object x
Find(x) returns representative object of the set containing x
Union(x,y) makes two sets containing x and y respectively into one set

Some Applications:
- Kruskal's algorithm for finding minimal spanning trees
- Finding connected components in graphs
- Finding connected components in images (binary)

This code is due to Ahmed El Deeb (http://code.activestate.com/recipes/577225-union-find/),
and is licensed under MIT License.
"""

def make_set(x):
     print(x)
     x.parent = x
     x.rank   = 0

def union(x, y):
     xRoot = find(x)
     yRoot = find(y)
     if xRoot.rank > yRoot.rank:
         yRoot.parent = xRoot
     elif xRoot.rank < yRoot.rank:
         xRoot.parent = yRoot
     elif xRoot != yRoot: # Unless x and y are already in same set, merge them
         yRoot.parent = xRoot
         xRoot.rank = xRoot.rank + 1

def find(x):
     if x.parent == x:
        return x
     else:
        x.parent = Find(x.parent)
        return x.parent