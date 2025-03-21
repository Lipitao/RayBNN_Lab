function y = UAF(x,UAF_A,UAF_B,UAF_C,UAF_D,UAF_E)
P1 = (UAF_A .* (x+UAF_B)) + (UAF_C .* (x .^ 2.0)) ;
P2 = (UAF_D .* (x-UAF_B)) ;

P1 = max(0, P1) + log1p(exp(-abs(P1)));
P2 = max(0, P2) + log1p(exp(-abs(P2)));
y = P1 - P2 + UAF_E;
end
