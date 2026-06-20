function reproduce_paraflux_plot (mbm_data)
    
    if nargin < 1
        [filename, filepath] = uigetfile({'*.mat'}, 'MBM file');
        if isequal(filename, 0)
            return;
        end
        mbm_data = load(fullfile(filepath, filename));
    end


    tid = mbm_data.gri;
    xyz = mbm_data.xyz;
    tim = mbm_data.tim;

    uid = unique(tid);
    nLoc = arrayfun(@(id) sum(tid==id), uid);
    
    uid( nLoc<10 ) = [];
    nCol = length(uid);

    figure;
    
    for i = 1 : nCol
        
        tim_i = tim(tid==uid(i));
        xyz_i = 1e9* xyz(tid==uid(i), :);
        xyz_i = xyz_i - mean(xyz_i);

        subplot(4, nCol, i);
        plot(xyz_i(:,1), xyz_i(:,2), '-o', 'Color', [1, 0.65, 0], 'MarkerSize', 3);
        xlim([-50, 300]); ylim([-50, 70]); 
        %axis equal;
        xlabel("x-drift/nm"); ylabel("y-drift/nm");
        title( num2str(uid(i)) );
        
        subplot(4, nCol, i+nCol);
        plot(tim_i, xyz_i(:, 1), '-o', 'Color', [1, 0.65, 0], 'MarkerSize', 3);
        ylim([-50, 300]);
        ylabel("x-drift/nm");

        subplot(4, nCol, i+2*nCol);
        plot(tim_i, xyz_i(:, 2), '-o', 'Color', [1, 0.65, 0], 'MarkerSize', 3);
        ylim([-50, 400]);
        ylabel("y-drift/nm");

        subplot(4, nCol, i+3*nCol);
        plot(tim_i, xyz_i(:, 3), '-o', 'Color', [1, 0.65, 0], 'MarkerSize', 3);
        ylim([-90, 110]);
        ylabel("z-drift/nm");
        xlabel("time/s");


    end
    
    


end