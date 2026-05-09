% 2D Iterative Image Reconstruction Using the Adjoint Example
%
% github.com/ucl-bug/k-wave/blob/main/k-Wave/examples/example_pr_2D_adjoint.m
%
% This example demonstrates how the adjoint operator to the photoacoustic
% forward model can be used in an iterative image reconstruction based on
% gradient descent. This data-matching approach is the simplest case of a
% class of minimisation-based approaches that can include prior information
% in the form of additional penalty functionals. The example is very
% similar to 2D Iterative Image Improvement Using Time Reversal Example, as
% the techqniues are closely related.  
%
% For a more detailed discussion of this example and the underlying
% techniques, see Arridge et al. Inverse Problems 32, 115012 (2016).   
%
% author: Ben Cox, Bradley Treeby and Felix Lucka
% date: 29th May 2017
% last update: 22nd October 2021
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2017- Ben Cox and Bradley Treeby

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
%
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

clearvars;

% =========================================================================
% SET UP AND RUN THE SIMULATION
% =========================================================================

% define literals
NUMBER_OF_ITERATIONS = 5;  % number of iterations
PML_SIZE = 20;              % size of the perfectly matched layer in grid points
RUN_ADJOINT_TEST = true;    % run a dot-product test for the numerical adjoint
EXPORT_REFERENCE_H5 = true;
REFERENCE_H5_PATH = fullfile('..', 'test-data', 'kWave_results_2.h5');

% create the computational grid
Nx = 128 - 2 * PML_SIZE;    % number of grid points in the x direction
Ny = 256 - 2 * PML_SIZE;    % number of grid points in the y direction
dx = 0.1e-3;                % grid point spacing in the x direction [m]
dy = 0.1e-3;                % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;	% [m/s]

% create the time array
kgrid.makeTime(medium.sound_speed);

% load an image for the initial pressure distribution
p0_image = loadImage('EXAMPLE_k-Wave.png');

% make it binary
p0_image = double(p0_image > 0);

% smooth and scale the initial pressure distribution
p0 = smooth(p0_image, true);

% assign to the source structure
source.p0 = p0;

% define an L-shaped array
sensor.mask = zeros(Nx, Ny);
sensor.mask(1, :) = 1;
sensor.mask(:, 1) = 1;

% set the input arguments: force the PML to be outside the computational
% grid, switch off p0 smoothing within kspaceFirstOrder2D
input_args = {'PMLInside', false, 'PMLSize', PML_SIZE, 'Smooth', false, ...
    'PlotPML', false, 'PlotSim', false};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% =========================================================================
% EXPORT MATLAB REFERENCE DATA FOR PYTEST
% =========================================================================

if EXPORT_REFERENCE_H5
    % Time reversal reference image.
    source_tr = struct();
    source_tr.p_mask = sensor.mask;
    source_tr.p_mode = 'dirichlet';
    source_tr.p = sensor_data(:, end:-1:1);

    sensor_image = struct();
    sensor_image.mask = ones(Nx, Ny);
    sensor_image.record = {'p_final'};

    tr_ref = kspaceFirstOrder2D(kgrid, medium, source_tr, sensor_image, input_args{:});

    % Adjoint reference image (updated MATLAB formula).
    source_adj_ref = struct();
    source_adj_ref.p_mask = sensor.mask;
    source_adj_ref.p_mode = 'additive';
    p_adj_ref           = [sensor_data(:, end:-1:1), zeros(size(sensor_data, 1), 1)] ...
                        + [zeros(size(sensor_data, 1), 1), sensor_data(:, end:-1:1)];
    p_adj_ref(:, end-1) = p_adj_ref(:, end-1) + p_adj_ref(:, end);
    p_adj_ref           = p_adj_ref(:, 1:end-1);
    source_adj_ref.p = p_adj_ref;

    adj_ref = kspaceFirstOrder2D(kgrid, medium, source_adj_ref, sensor_image, input_args{:});

    % Keep Python test sign convention unchanged: tests compare against
    % negatives of the raw p_final images.
    tr_image = -tr_ref.p_final;
    adj_image = -adj_ref.p_final;

    % Store sensor data in (Nt, Ns) layout to match Python tests.
    data_nt_ns = sensor_data.';

    if exist(REFERENCE_H5_PATH, 'file')
        delete(REFERENCE_H5_PATH);
    end

    h5create(REFERENCE_H5_PATH, '/p0', size(p0), 'Datatype', 'double');
    h5write(REFERENCE_H5_PATH, '/p0', p0);

    h5create(REFERENCE_H5_PATH, '/data', size(data_nt_ns), 'Datatype', 'double');
    h5write(REFERENCE_H5_PATH, '/data', data_nt_ns);

    h5create(REFERENCE_H5_PATH, '/tr_image', size(tr_image), 'Datatype', 'double');
    h5write(REFERENCE_H5_PATH, '/tr_image', tr_image);

    h5create(REFERENCE_H5_PATH, '/adj_image', size(adj_image), 'Datatype', 'double');
    h5write(REFERENCE_H5_PATH, '/adj_image', adj_image);

    created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
    h5writeatt(REFERENCE_H5_PATH, '/', 'generator_script', 'tests/mat-file/example_pr_2D_adjoint_vs_TR.m');
    h5writeatt(REFERENCE_H5_PATH, '/', 'adjoint_formula', '[r,0] + [0,r], fold end into end-1');
    h5writeatt(REFERENCE_H5_PATH, '/', 'data_layout', 'data is (Nt, Ns)');
    h5writeatt(REFERENCE_H5_PATH, '/', 'created_utc', created_utc);
    h5writeatt(REFERENCE_H5_PATH, '/', 'kwave_version', 'unknown');
end

% =========================================================================
% OPTIONAL: NUMERICAL ADJOINT CHECK (DOT-PRODUCT TEST)
% =========================================================================

if RUN_ADJOINT_TEST
    % The discrete adjoint should satisfy <A x, y>_D ~= <x, A^T y>_X where
    % <.,.>_D includes the time step and <.,.>_X includes the grid spacing.
    % Expect a small but non-zero mismatch due to PML, finite time window,
    % and other discretization effects; smoothing/positivity are excluded.
    rng(0);
    p0_test = rand(Nx, Ny);
    source_test = source;
    source_test.p0 = p0_test;
    sensor_test = sensor;

    Ax = kspaceFirstOrder2D(kgrid, medium, source_test, sensor_test, input_args{:});
    data_test = randn(size(Ax));

    source_adj = struct();
    source_adj.p_mask = sensor.mask;
    source_adj.p_mode = 'additive';
    p_adj_test           = [data_test(:, end:-1:1), zeros(size(data_test, 1), 1)] ...
                         + [zeros(size(data_test, 1), 1), data_test(:, end:-1:1)];
    p_adj_test(:, end-1) = p_adj_test(:, end-1) + p_adj_test(:, end);
    p_adj_test           = p_adj_test(:, 1:end-1);
    source_adj.p = p_adj_test;

    sensor_adj = sensor;
    sensor_adj.record = {'p_final'};

    Aty = kspaceFirstOrder2D(kgrid, medium, source_adj, sensor_adj, input_args{:});
    Aty = Aty.p_final;

    lhs = sum(Ax(:) .* data_test(:)) * kgrid.dt;
    rhs = sum(p0_test(:) .* Aty(:)) * dx * dy;
    adjoint_rel_err = abs(lhs - rhs) / max([abs(lhs), abs(rhs), eps]);
    fprintf('Adjoint dot-product test: rel err = %.3e\n', adjoint_rel_err);
end

% =========================================================================
% RECONSTRUCT THE IMAGE USING A GRADIENT DESCENT ALGORITHM
% =========================================================================

% set the initial reconstructed image to be zeros
p0_estimate = zeros(Nx, Ny);

% set the initial model times series to be zero
modelled_time_series = zeros(size(sensor_data));

% calculate the difference between the measured and modelled data
data_difference = modelled_time_series - sensor_data;

% track goodness of fit and relative error during the iteration
gof     = ones(NUMBER_OF_ITERATIONS+1, 1);
rel_err = ones(NUMBER_OF_ITERATIONS+1, 1); 

% reconstruct p0 image iteratively using the adjoint to estimate the
% functional gradient
for loop = 1:NUMBER_OF_ITERATIONS

    % remove the initial pressure used in the forward simulation
    source = rmfield(source, 'p0');
    
    % make the source points for the adjoint the same as the sensor points
    % for the forward model
    source.p_mask = sensor.mask;
    
    % set the simulation to record the final image (at t = 0)
    sensor.record = {'p_final'};
    
    % set the source type to act as an adjoint source
    source.p_mode = 'additive';
    
    % assign the difference time series as an adjoint source
    % (see Appendix B, Eqn B.2 in Arridge et al. Inverse Problems 32, 115012 (2016))
    p_adj           = [data_difference(:, end:-1:1), zeros(size(data_difference, 1), 1)] ...
                    + [zeros(size(data_difference, 1), 1), data_difference(:, end:-1:1)];
    p_adj(:, end-1) = p_adj(:, end-1) + p_adj(:, end);
    p_adj           = p_adj(:, 1:end-1);
    source.p = p_adj;
    
    % send difference through adjoint model
    image_update = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    
    % add smoothing (see Appendix B in Arridge et al. Inverse Problems 32, 115012 (2016))
    image_update = smooth(image_update.p_final, false);
    
    % set the step length (this could be chosen by doing a line search)
    nu = 0.5;
    
    % update the image    
    p0_estimate = p0_estimate - nu * image_update;
    
    % apply a positivity condition
    p0_estimate = p0_estimate .* (p0_estimate >= 0);
    
    % store the latest image estimate
    p0_iterates{loop} = p0_estimate;

    % set the latest image to be the initial pressure in the forward model
    source = rmfield(source, 'p');
    source.p0 = smooth(p0_estimate, true);
    
    % set the sensor to record time series (by default)
    sensor = rmfield(sensor, 'record');
    
    % calculate the time series at the sensors using the latest estimate of p0
    modelled_time_series = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    
    % calculate the difference between the measured and modelled data
    data_difference = modelled_time_series - sensor_data;
       
    % measure goodness of fit and relative error
    gof(loop+1)     = norm(data_difference(:))^2/norm(sensor_data(:))^2;
    rel_err(loop+1) = norm(p0_estimate(:) - p0(:))^2/norm(p0(:))^2;    
        
end

% =========================================================================
% VISUALISATION
% =========================================================================

% set the color scale
c_axis = [0, 1.1];

% plot the initial pressure
figure;
imagesc(p0, c_axis);
axis image;
set(gca, 'XTick', [], 'YTick', []);
title('Initial Acoustic Pressure');
colorbar;
scaleFig(1, 0.65);

% add lines indicating the sensor mask
hold on;
plot([Ny, 1], [1, 1], 'k-');
plot([1, 1], [1, Nx], 'k-');

% plot the first iteration
figure;
imagesc(p0_iterates{1}, c_axis);
axis image;
set(gca, 'XTick', [], 'YTick', []);
title('Gradient Descent, 1 Iteration');
colorbar;
scaleFig(1, 0.65);

% add lines indicating the sensor mask
hold on;
plot([Ny, 1], [1, 1], 'k-');
plot([1, 1], [1, Nx], 'k-');

% plot the 2nd iteration
figure;
imagesc(p0_iterates{2}, c_axis);
axis image;
set(gca, 'XTick', [], 'YTick', []);
title('Gradient Descent, 2 Iterations');
colorbar;
scaleFig(1, 0.65);

% add lines indicating the sensor mask
hold on;
plot([Ny, 1], [1, 1], 'k-');
plot([1, 1], [1, Nx], 'k-');

% plot the last iteration
figure;
imagesc(p0_iterates{end}, c_axis);
axis image;
set(gca, 'XTick', [], 'YTick', []);
title(['Gradient Descent, ' int2str(NUMBER_OF_ITERATIONS) ' Iterations']);
colorbar;
scaleFig(1, 0.65);

% add lines indicating the sensor mask and 'visible region'
hold on;
plot([Ny, 1], [1, 1], 'k-');
plot([1, 1], [1, Nx], 'k-');
plot([Ny, 1], [1, Nx], 'k--');

% plot gof and relative error
figure();
plot(0:NUMBER_OF_ITERATIONS, gof, 'DisplayName', 'goodness of fit'); hold on
plot(0:NUMBER_OF_ITERATIONS, rel_err, 'DisplayName', 'relative error'); hold on
legend(gca,'show');
