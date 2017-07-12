function [perf_mse, perf_r2] = evaluate(t, y, net)
perf_mse = mse(net,t,y);
perf_r2 = 0;

end

