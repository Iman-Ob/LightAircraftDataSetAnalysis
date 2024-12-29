function y= fetchAnomal(pred,act)
    mu = mean(act);
    sig = std(act);
    actstd = (act - mu) ./ sig;

    rmse = sqrt(mean((pred-actstd').^2));
    anomal=(rmse> 0.1);
    if (anomal==1)
        y=1;
    else
        y=0;
    end
end
