
(cl:in-package :asdf)

(defsystem "map_manager-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "CheckPosCollision" :depends-on ("_package_CheckPosCollision"))
    (:file "_package_CheckPosCollision" :depends-on ("_package"))
  ))