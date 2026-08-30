function Pareto = fun1(obj)
Obj = obj;

[FrontNo,MaxFNo] = NDSort(Obj,inf);

j = 1; k = 1;
for i = 1:size(Obj)
    if FrontNo(i)==1
        ndo(j)=i;
        j = j + 1;
    else
        do(k) = i;
        k = k + 1;
    end
end
Pareto = Obj(ndo,:);
remain = Obj(do,:);