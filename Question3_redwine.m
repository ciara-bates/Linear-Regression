load('red_wine_quality_data.mat', 'red_wine_x');
load('red_wine_quality_data.mat', 'red_wine_y');

rng(1); % random seed (randomises the same each time when seed = 1)
rand_pos = randperm(length(red_wine_y)); %array of random positions 1 to 1599

% pre-defining the structure of the shuffled data
shuffled_red_wine_x = zeros(1599,11);
shuffled_red_wine_y = zeros(1599,1);

% looping through 1 to 1599 to create new arrays with original data randomly distributed 
for k = 1:length(red_wine_y)
    shuffled_red_wine_x(k,:) = red_wine_x(rand_pos(k),:);% entire row of x values
    shuffled_red_wine_y(k) = red_wine_y(rand_pos(k));
end

% calculating the number of samples in the test dataset (75-25 split)
test_length = round(0.25*length(red_wine_y));

% split shuffled datasets into test and training datasets
test_x = shuffled_red_wine_x(1:test_length,:); % first 400 samples
test_y = shuffled_red_wine_y(1:test_length);

train_x = shuffled_red_wine_x(test_length+1:end,:); % 401 to 1599 samples
train_y = shuffled_red_wine_y(test_length+1:end);

% standardising data
[Xnew, PS] = mapstd(test_x'); % must transform test_x   
test_x_std = (mapstd('apply', test_x', PS))';

[Ynew, PS] = mapstd(test_y');   
test_y_std = (mapstd('apply', test_y', PS))';

[Xnew, PS] = mapstd(train_x');  
train_x_std = (mapstd('apply', train_x', PS))';

[Ynew, PS] = mapstd(train_y');   
train_y_std = (mapstd('apply', train_y', PS))';


% display the first 5 rows of the test and train datasets 
disp('test dataset (standardised)')
test_y_std(1:5)
test_x_std(1:5,:)

disp('train dataset (standardised)')
train_y_std(1:5)
train_x_std(1:5,:)

% linear regression model ('estimate' = beta values)
mdl = fitlm(train_x_std,train_y_std)

% Table of feature coefficents sorted by magnitude of beta values
coefficients_table = sortrows(mdl.Coefficients,'Estimate', {'descend'}, 'ComparisonMethod','abs') 






