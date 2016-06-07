import sys
import vtk

if all(f in dir(vtk) for f in ['vtkImageExport', 'vtkStructuredPointsReader','VTKtoNumpy']):
    sys.exit(1)
