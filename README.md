# Algorithms-For-Evolving-Shapes
Given a sequence of shapes registers (matches) the shapes at different times and analyzes the changes in the metric with the curvature of the surface


Second Commit:
Continuum morphometry describes a formalisms that extendsa kind of quasiconformal transformations to a setting of surfaces in embeddede in 3D. The PDF describes both the problem of matching two nearly identical surface, which is relavant to growing surfaces. And the problem of registering different surfaces. The distance given in this case includes elastic distortions in the process of matching in addition to gradients in those distortions, which has not been considered before.

The second PDF talks about the embedding of a 2D elastic mesh given the metric you want to embed. For example, if you compute your metric using a square root velocity function as described in the book "Functional and Shape Data Analysis", this code can embedd that metric optimally in 2D.


Third Commit: 
Run the script "GrowthProcesses.py" and call the method "scaled_hexagonal_lattice" to get the example shown in "anim.gif"

Add your own "metric_function" and call the method growth to obtain your own results.

The mathemtica notebook will help you to visualize the results.


Fourth Commit: 
Example mesh information have been added to the folder "Mesh_Data". This contains things like Gaussian curvature, normals, faces, principal directions and so on. These are needed in the computation of registrations. 

Future versions will contain code that will calculated these quantities, given two mesh as list of vertices and edges.
