Submit one ZIP through the university information system. Its root must contain:

  best_model/
    my_model.py
    model.pkl
    utils.py
  training_code/
    train.py
    <other files used in training>
  declarations/
    group_declaration_of_honor.pdf
    student_1_ai_declaration.pdf
    student_2_ai_declaration.pdf

The university information system identifies the group from the submission
folder name. No manifest or group-identifier file is required.

The supplied files in best_model/ implement a dummy model that always predicts
zero. They are included only to demonstrate a structurally valid interface and
to let students run the submission simulation immediately. Replace my_model.py, model.pkl,
and utils.py with the group's fitted model and prediction implementation before
submission. The dummy model has no meaningful predictive performance.

The grader imports best_model/my_model.py, with best_model/ available for
sibling imports from utils.py. It calls load_model(best_model/model.pkl) once,
then calls predict(model, history_df, day_features_df) once per 2024 day.
predict must return a finite NumPy array of shape (1,). It must not read files,
use the network, retrain, modify its inputs, or rely on the working directory.

Tools from sklearn.preprocessing may be used to transform input features.
Fit any preprocessing that learns parameters on training data only, and apply
the same fitted transformation to validation and prediction inputs in the
same feature order. Lag construction, target transformation, and output
post-processing must remain explicit in the submitted Python source. Library
predictive estimators such as Ridge or LASSO are allowed. Library pipelines,
composition wrappers, imputers, feature selectors, decomposition objects,
and output-calibration objects remain forbidden.

model.pkl may contain the fitted predictive estimator, fitted
sklearn.preprocessing objects for input features, and plain state such as
dictionaries, lists, strings, numbers, and NumPy arrays. A fitted scaler may
store its learned means and scales; save other required quantities as plain
values. Do not serialize a Pipeline, ColumnTransformer,
TransformedTargetRegressor, or other prohibited wrapper. The submission
simulation checks both submitted Python imports and the loaded artifact.

The training_code directory must contain the Python source that reproduces
data loading, explicit lag construction and preprocessing, model training, and
creation of best_model/model.pkl. Replace the supplied train.py template.

The declarations directory must contain the three exactly named PDF files shown
above. The order of student_1 and student_2 must match the order in the group
declaration. Editable Word templates are provided in the sibling
declaration_templates/ directory of the student package. Complete and sign the
templates, export them to PDF, and place only the resulting PDFs here. The
submission simulation checks that the files are present and valid PDFs.

Run `validate_submission.py` as a submission simulation before creating the ZIP.

The hidden test reports R2, RMSE, and MAE. R2 is the primary assessment metric;
RMSE and MAE are secondary diagnostics.
