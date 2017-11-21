
How the hell is this going to work?

collision
 * multiple collision layers

(water, rock, fence)

terrain
 * limit one terrain per tile
 * collidable, but can only be one collidability type?

token
 * tokens can share a tile
 * collidable
 *

simulant

actor


Assuming we HAVE a universe loaded

every tick, we run a step of universe time

then, we take the list of changes to the universe and both
* use them to update the Redis state
* push the changes themselves to anybody who's subscribed to those changes

one of the things that we need in order for this to make sense is a
serializable delta object that represents a change to world state



A "Player" object is a collection of Actors that the player can send commands to,
as well as a collection of Views that the player can see.

* there's a special protocol to dump a whole BUNCH of keys into Redis at once

we're going to avoid doing that, right?


Okay, so, the Terrain Map
is that backed by redis?
do we generate a unique ID for each Terrain Map?
do we sync the terrain map separately from the rest of things


Can we keep all of the visual details out of the backend?
A terrain might have a name, but not a texture.
