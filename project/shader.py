import logging
import torch
import vector

from light import Light
from svbrdf import SVBRDF
from torch import Tensor
from typing import Sequence
from viewer import Viewer


def shade(surface: Tensor, normals: Tensor, lights: Sequence[Light], viewer: Viewer, svbrdf: SVBRDF) -> Tensor:
    '''
    Evaluates the rendering equation for the given surface with respect to the provided normals, Lights, Viewer, and SVBRDF.

    Args:
        surface: Tensor [R, C, 3] of points.
        normals: Tensor [B, R, C, 3] of surface normals.
        lights: Lights illuminating the surface.
        viewer: Viewer observing the surface.
        svbrdf: SVBRDF of the surface.

    Returns:
        Tensor [B, R, C, 3] of the exitant radiance from each point on the surface.
    '''
    assert len(surface.shape) == 3, 'Surface must have 3 dimensions.'
    assert surface.size(2) == 3, 'Surface must have an (X, Y, Z) value for each spatial parameter.'
    assert len(normals.shape) == 4, 'Normals must have 4 dimensions.'
    assert normals.size(3) == 3, 'Normals must have an (X, Y, Z) value for each spatial parameter.'
    outbound_radiance = torch.zeros_like(normals)
    for light in lights:
        # Gather the incident and outbound directions; multiple incident directions may be associated with each point.
        incident_directions = light.directions(surface).unsqueeze(0).expand(normals.size(0), -1, -1, -1, -1)
        outbound_directions = viewer.directions(surface).unsqueeze(0).unsqueeze(3).expand(incident_directions.shape)
        # Expand the surface normal Tensor to match the shape of the incident and outbound direction Tenors.
        expanded_normals = normals.unsqueeze(3).expand(incident_directions.shape)
        # Compute the incident radiosity on each point on the surface, taking care to clamp away negative cosine values.
        incident_radiosity = light.radiance(surface).unsqueeze(0) * vector.dot(incident_directions, expanded_normals).clamp(0, 1)
        # For every point on the surface, evaluate the SVBRDF for each normal, incident, and outbound direction triplet.
        svbrdf_eval_results = svbrdf.evaluate(expanded_normals, incident_directions, outbound_directions)
        # Sum the radiance values along each normal, incident, and outbound direction triplet to compute the desired integral.
        outbound_radiance += (incident_radiosity * svbrdf_eval_results).sum(3)
    logging.debug('Shaded %dx%d texels from %d light(s)', surface.size(1), surface.size(0), len(lights))
    return outbound_radiance.clamp(0, 1)
